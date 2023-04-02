from src.run_tracker import run_tracker
from src.correlation_filter_tracker import CorelationParams
sequence = "cup"
parameters = CorelationParams()
fps, failures = run_tracker("./data", sequence, parameters)

print(fps)
print(failures)