import time
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
import robosuite as suite
from robosuite.wrappers import GymWrapper


class SmolVLMAgent:
    """
    SmolVLM-based agent for cube lifting task.
    Follows the same structure as TD3 agent but uses vision-language model.
    """
    
    def __init__(
        self,
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        device=None,
        camera_names=None,
    ):
        """
        Initialize SmolVLM agent.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading SmolVLM model: {model_name} on {self.device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        
        print("SmolVLM model loaded successfully!")
        
        # Track state
        self.step_count = 0
        self.phase = "approach"  # approach, grasp, lift
        # List of camera names to use for visual input. If multiple names are
        # provided, images will be stitched side-by-side and sent to the VLM.
        # Replace these names with camera names available in your MuJoCo model.
        if camera_names is None:
            self.camera_names = ["frontview"]
        else:
            self.camera_names = list(camera_names)
        
    def get_camera_image(self, env):
        """
        Get RGB image from environment camera.
        
        Args:
            env: Robosuite environment (wrapped or unwrapped)
            
        Returns:
            PIL Image
        """
        # Render each requested camera and return a single PIL Image. If more
        # than one camera is provided, stitch images horizontally.
        imgs = []
        w, h = 512, 512

        for cam in self.camera_names:
            try:
                # Prefer direct sim.render for deterministic camera selection
                cam_obs = env.env.sim.render(
                    width=w,
                    height=h,
                    camera_name=cam,
                )
                img = Image.fromarray(cam_obs)
                imgs.append(img)
            except Exception as e:
                # Fall back to env.render if direct sim.render fails for some wrappers
                try:
                    # env.render may accept camera_name depending on robosuite version
                    fallback = env.render(camera_name=cam)
                    if isinstance(fallback, np.ndarray):
                        imgs.append(Image.fromarray(fallback))
                    elif isinstance(fallback, Image.Image):
                        imgs.append(fallback)
                    else:
                        # Last resort: create a blank image placeholder
                        print(f"Warning: could not render camera '{cam}': {e}")
                        imgs.append(Image.new('RGB', (w, h), color=(128, 128, 128)))
                except Exception:
                    print(f"Warning: could not render camera '{cam}': {e} - inserting blank image")
                    imgs.append(Image.new('RGB', (w, h), color=(128, 128, 128)))

        if len(imgs) == 0:
            # Shouldn't happen, but provide a gray image if all rendering failed
            return Image.new('RGB', (w, h), color=(128, 128, 128))
        elif len(imgs) == 1:
            return imgs[0]
        else:
            # Stitch images side-by-side
            total_w = w * len(imgs)
            stitched = Image.new('RGB', (total_w, h))
            for i, im in enumerate(imgs):
                # Ensure each image matches expected size
                if im.size != (w, h):
                    im = im.resize((w, h))
                stitched.paste(im, (i * w, 0))
            return stitched
    
    def create_prompt(self, observation, action_space_size):
        """
        Create prompt for SmolVLM based on current state.
        
        Args:
            observation: Current state observation
            action_space_size: Size of action space
            
        Returns:
            Formatted prompt string
        """
        # For Lift environment with OSC_POSE controller
        # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        
        prompt = f"""<|im_start|>system
You are a robot control assistant for a Panda robot arm that needs to pick up a cube and lift it.

Current Status:
- Step: {self.step_count}
- Current Phase: {self.phase}
- Action space: {action_space_size} dimensions with OSC_POSE controller
  [dx, dy, dz, droll, dpitch, dyaw, gripper]
  - dx, dy, dz: change in end-effector position (meters), typically -0.05 to 0.05
  - droll, dpitch, dyaw: change in orientation (radians), typically -0.1 to 0.1
  - gripper: -1 (open) to 1 (close)

Task: Pick up the cube and lift it above the table.

Analyze the image and provide a control action as a comma-separated list of {action_space_size} numbers.

Response Format:
Provide ONLY a comma-separated list of numbers like: 0.02, 0.01, -0.01, 0.0, 0.0, 0.0, -1.0

Guidelines based on phase:
- APPROACH phase: Move end-effector toward cube (adjust dx, dy, dz), keep gripper open (-1)
- GRASP phase: Lower to cube level (negative dz), then close gripper (1)
- LIFT phase: Move upward (positive dz around 0.05), keep gripper closed (1)
<|im_end|>
<|im_start|>user
What action should the robot take now?
<image>
<|im_end|>
<|im_start|>assistant
Action: """
        
        return prompt
    
    def parse_action(self, response, n_actions):
        """
        Parse action from model response.
        
        Args:
            response: Text response from model
            n_actions: Expected number of action dimensions
            
        Returns:
            Action array
        """
        try:
            # Look for comma-separated numbers in response
            import re
            
            # Try to find comma-separated numbers
            number_pattern = r'[-+]?\d*\.?\d+'
            matches = re.findall(number_pattern, response)
            
            if len(matches) >= n_actions:
                # Convert to float and take first n_actions
                action = np.array([float(m) for m in matches[:n_actions]], dtype=np.float32)
                # Clip to valid range
                action = np.clip(action, -1.0, 1.0)
                return action
            else:
                # Fallback: generate simple heuristic action
                return self.get_heuristic_action(n_actions)
                
        except Exception as e:
            print(f"Error parsing action: {e}")
            print(f"Response was: {response}")
            return self.get_heuristic_action(n_actions)
    
    def get_heuristic_action(self, n_actions):
        """
        Generate heuristic action based on current phase for cube lifting.
        
        Args:
            n_actions: Action space size (7 for OSC_POSE)
            
        Returns:
            Action array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action = np.zeros(n_actions, dtype=np.float32)
        
        if self.phase == "approach":
            # Move forward and down toward cube
            action[0] = 0.02   # Move forward slightly
            action[1] = 0.0    # No lateral movement
            action[2] = -0.02  # Move down toward table
            action[6] = -1.0   # Keep gripper open
            
        elif self.phase == "grasp":
            # Move down and close gripper
            action[2] = -0.01  # Continue moving down
            action[6] = 1.0    # Close gripper
            
        elif self.phase == "lift":
            # Lift cube upward
            action[2] = 0.05   # Move up
            action[6] = 1.0    # Keep gripper closed
            
        return action
    
    def update_phase(self, observation):
        """
        Update task phase based on step count and observation.
        
        Args:
            observation: Current observation
        """
        if self.step_count < 30:
            self.phase = "approach"
        elif self.step_count < 50:
            self.phase = "grasp"
        else:
            self.phase = "lift"
    
    def choose_action(self, env, observation, validation=True):
        """
        Choose action using SmolVLM inference.
        
        Args:
            env: Environment instance
            observation: Current state observation
            validation: If True, run in validation mode (same as TD3)
            
        Returns:
            Action array
        """
        self.step_count += 1
        self.update_phase(observation)
        
        # Get camera image
        image = self.get_camera_image(env)
        # Determine expected action dimension from the environment when possible
        try:
            env_action_dim = int(env.action_space.shape[0])
        except Exception:
            env_action_dim = None

        # Create prompt
        n_actions = env_action_dim if env_action_dim is not None else (len(observation) if hasattr(observation, '__len__') else 7)
        prompt = self.create_prompt(observation, n_actions)
        
        # Prepare inputs for model
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,  # Lower temperature for more consistent actions
                do_sample=True,
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse action from response (may produce more/less entries)
        action = self.parse_action(response, n_actions)

        # Ensure action matches environment's expected dimension
        if env_action_dim is not None:
            try:
                if action.shape[0] > env_action_dim:
                    action = action[:env_action_dim]
                elif action.shape[0] < env_action_dim:
                    pad = np.zeros(env_action_dim, dtype=action.dtype)
                    pad[: action.shape[0]] = action
                    action = pad
            except Exception:
                # As a last resort, return zeros of correct shape
                action = np.zeros(env_action_dim, dtype=np.float32)
        
        # Print debug info occasionally
        if self.step_count % 10 == 0:
            print(f"\nStep {self.step_count} | Phase: {self.phase}")
            print(f"Response: {response[:150]}...")
            print(f"Action: {action}")
        
        return action
    
    def reset(self):
        """Reset agent state for new episode."""
        self.step_count = 0
        self.phase = "approach"


if __name__ == '__main__':
    
    # Create tmp directory if needed
    if not os.path.exists("tmp_smolvlm"):
        os.makedirs("tmp_smolvlm")
    
    env_name = "Lift"
    
    # Create environment - OSC_POSE controller for cube pickup
    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=True,
        use_camera_obs=False,  # We'll get camera via render instead
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )
    
    env = GymWrapper(env)
    
    print(f"\nEnvironment created successfully!")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Initialize SmolVLM agent
    # Example: use two cameras if your MJCF model exposes them (replace names as needed)
    requested_cameras = ["frontview", "sideview"]

    # Try to detect available camera names from the underlying MuJoCo model
    available_cams = None
    try:
        # env.env is the unwrapped robosuite environment; sim.model may expose camera names
        cam_names = []
        sim = getattr(env, 'env', None)
        if sim is not None and hasattr(sim, 'sim') and hasattr(sim.sim, 'model'):
            # Some robosuite/MuJoCo bindings expose cams via sim.model.cam_names or id mapping
            mj_model = sim.sim.model
            if hasattr(mj_model, 'cam_names'):
                cam_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in mj_model.cam_names]
        available_cams = cam_names
    except Exception:
        available_cams = None

    # Filter requested cameras against available cameras when possible
    if available_cams:
        selected_cameras = [c for c in requested_cameras if c in available_cams]
        missing = [c for c in requested_cameras if c not in available_cams]
        if missing:
            print(f"Warning: Requested cameras not found in model: {missing}. Using: {selected_cameras}")
        else:
            print(f"All requested cameras available: {selected_cameras}")
    else:
        # Could not auto-detect; use requested list and warn user
        selected_cameras = requested_cameras
        print(f"Could not detect available cameras programmatically; assuming: {selected_cameras}")

    agent = SmolVLMAgent(
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        camera_names=selected_cameras,
    )

    print(f"Using camera(s) for visual input: {agent.camera_names}")
    
    # Run episodes (same structure as test.py)
    n_games = 3
    best_score = 0

    # Run episodes with safe cleanup to avoid EGL/OpenGL destructor errors
    import atexit
    import gc

    def _cleanup(e=env):
        try:
            print("Cleaning up environment...")
            e.close()
        except Exception as e_close:
            print(f"Error while closing env during cleanup: {e_close}")

    # Ensure cleanup runs on normal exit, capture env in default arg
    atexit.register(_cleanup)

    try:
        for game in range(n_games):
            obs = env.reset()
            agent.reset()
            score = 0

            # Simple stepping loop; agent.choose_action expects the observation
            for t in range(env.horizon if hasattr(env, 'horizon') else 200):
                # Normalize observation if it's a dict (GymWrapper may return flat arrays or dicts)
                if isinstance(obs, dict):
                    # prefer keys named 'obs' or the first value
                    if 'obs' in obs:
                        observation = obs['obs']
                    else:
                        observation = next(iter(obs.values()))
                else:
                    observation = obs

                try:
                    action = agent.choose_action(env, observation)
                except Exception as e:
                    print(f"Agent failed to produce action at step {t}: {e}")
                    # fallback to random action of correct shape
                    try:
                        action = env.action_space.sample()
                    except Exception:
                        action = np.zeros(env.action_space.shape, dtype=np.float32)

                obs, reward, done, info = env.step(action)
                # accumulate scalar reward where possible
                try:
                    score += float(reward)
                except Exception:
                    pass

                # Try to render an on-screen window so the user can visualise the scene.
                # This is wrapped in try/except because in some headless setups render
                # may not be available or may raise EGL/OpenGL exceptions.
                try:
                    env.render()
                    # small pause so the window has time to update
                    time.sleep(0.02)
                except Exception as e:
                    # Don't spam logs; only show debug once per 100 steps
                    if t % 100 == 0:
                        print(f"env.render() failed (will continue headless): {e}")

                if done:
                    break

            best_score = max(best_score, score)
            print(f"Episode {game+1}/{n_games} score: {score:.3f} | best: {best_score:.3f}")

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Explicit cleanup before interpreter shutdown to avoid destructor errors
        try:
            print("Final cleanup: closing env and freeing resources...")
            env.close()
        except Exception as e:
            print(f"Error during final env.close(): {e}")

        try:
            del env
        except Exception:
            pass

        # force garbage collection and clear CUDA cache if available
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass