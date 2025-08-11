# Load required libraries and source files
source("FastStepGraph/R/FastStepGraph.R")

# Load the iris dataset
data(iris)
X <- iris[, 1:4]

# Run FastStepGraph
g <- FastStepGraph(X, alpha_f = 0.22, alpha_b = 0.14, data_scale=TRUE, nei.max=3L)

# Save the data and results to CSV files
write.csv(X, "tests/fixtures/data/iris_X.csv", row.names = FALSE)
write.csv(g$Omega, "tests/fixtures/data/iris_omega.csv", row.names = FALSE)
write.csv(g$beta, "tests/fixtures/data/iris_beta.csv", row.names = FALSE)
write.csv(g$Edges, "tests/fixtures/data/iris_edges.csv", row.names = FALSE)

print("Iris test data generated successfully.") 