# Load required libraries and source files
source("FastStepGraph/R/SigmaAR.R")
source("FastStepGraph/R/FastStepGraph.R")

# Generate synthetic data
set.seed(0)
data <- SigmaAR(100, 20, 0.4)
X <- data$X

# Run FastStepGraph
g <- FastStepGraph(X, alpha_f = 0.22, alpha_b = 0.14, data_scale=TRUE)

# Save the data and results to CSV files
write.csv(X, "tests/data/X.csv", row.names = FALSE)
write.csv(g$Omega, "tests/data/omega.csv", row.names = FALSE)
write.csv(g$beta, "tests/data/beta.csv", row.names = FALSE)
write.csv(g$Edges, "tests/data/edges.csv", row.names = FALSE)

print("Test data generated successfully.") 