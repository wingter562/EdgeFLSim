import argparse
import pandas as pd
from config import Config
from simulation.runner import run
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Edge Intelligence Simulation System")
    parser.add_argument("--num_devices", type=int, default=5, help="Number of edge devices")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--batch_size", type=int, default=256, help="Local batch size")
    parser.add_argument("--selection_strategy", type=str, default="hybrid",
                        choices=["random", "energy_aware", "capability_aware", "hybrid"])
    parser.add_argument("--aggregation_method", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "fedavg_energy", "qfed", "capability_weighted"])
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--compare", action="store_true", help="Run comparison for 5 and 8 devices")
    args = parser.parse_args()

    if args.compare:
        # 运行5台和8台设备的对比实验
        devices_list = [5, 8]
        all_results = []
        for nd in devices_list:
            print(f"\n{'#'*60}")
            print(f"Running simulation with {nd} devices")
            print(f"{'#'*60}")
            config = Config()
            config.num_devices = nd
            config.num_rounds = args.rounds
            config.batch_size = args.batch_size
            config.selection_strategy = args.selection_strategy
            config.aggregation_method = args.aggregation_method
            if args.no_cuda:
                config.use_cuda = False
            save_dir = f"{args.save_dir}_{nd}devices"
            df = run(config, save_dir=save_dir)
            final_acc = df['accuracy'].iloc[-1]
            total_energy = df['total_energy'].sum()
            all_results.append({
                'num_devices': nd,
                'final_accuracy': final_acc,
                'total_energy': total_energy
            })
        # 输出对比结果
        comp_df = pd.DataFrame(all_results)
        comp_df.to_csv(f"{args.save_dir}_comparison.csv", index=False)
        print("\nComparison Results:")
        print(comp_df)
    else:
        # 单次运行
        config = Config()
        config.num_devices = args.num_devices
        config.num_rounds = args.rounds
        config.batch_size = args.batch_size
        config.selection_strategy = args.selection_strategy
        config.aggregation_method = args.aggregation_method
        if args.no_cuda:
            config.use_cuda = False
        df = run(config, save_dir=args.save_dir)
        plt.plot(df["round"], df["accuracy"])
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.title("Federated Learning Accuracy Curve")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()