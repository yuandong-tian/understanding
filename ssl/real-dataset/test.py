import hydra
import torch

@hydra.main(config_path="./config", config_name="test.yaml", version_base="1.1")
def main(args):
    print(f"Seed: {args.seed}, Cuda #device: {torch.cuda.device_count()}")

if __name__ == "__main__":
    main()
