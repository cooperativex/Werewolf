import argparse
import d3rlpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../dataset_process/gpt4_examples.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # fix seed
    d3rlpy.seed(args.seed)
    
    # load dataset
    with open(args.dataset, "rb") as f:
        dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

    # setup algorithm
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=5e-5,
        batch_size=32,
        alpha=4.0,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=32),
        n_critics=2,
        target_update_interval=1000,
    ).create(device=args.device)

    # calculate metrics
    td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes=dataset.episodes)

    # define interface for logging
    logger_adapter = d3rlpy.logging.CombineAdapterFactory([
        d3rlpy.logging.FileAdapterFactory(root_dir="file_logs"),
        d3rlpy.logging.TensorboardAdapterFactory(root_dir="tensorboard_logs")
    ])

    cql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=5000,
        evaluators={'td_error': td_error_evaluator},
        experiment_name=f"DiscreteCQL_{args.seed}",
        logger_adapter=logger_adapter
    )


if __name__ == "__main__":
    main()
