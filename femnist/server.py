from typing import List, Tuple

import flwr.server.strategy
from flwr.server.client_manager import SimpleClientManager
# from fedavg_local import FedAvg
import flwr as fl
from flwr.common import Metrics

from client import client_fn




def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print("accuracy : {}".format(sum(accuracies) / sum(examples)))
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy
# client_manager = SimpleClientManager()
strategy = flwr.server.strategy.FedAvg(min_fit_clients=3, min_available_clients=3, evaluate_metrics_aggregation_fn=weighted_average)
# server = Server(client_manager=client_manager, strategy=strategy)

# Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=10),
#     strategy=strategy,
# )

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=10,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
