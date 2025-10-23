from pepinn import RadiativeTransferNN1D, IncidentBoundaryPINN1D, PEPINNTrainer
from pepinn import ExperimentConfig, TrainingConfig, ModelConfig, SolverConfig

# Create configuration
config = ExperimentConfig(
    name="test_case1",
    description="1D Incident Boundary",
    training=TrainingConfig(n_epochs=5000),
    model=ModelConfig(hidden_dim=128, n_layers=5),
    solver=SolverConfig(n_gauss=16)
)

# Initialize model and solver
model = RadiativeTransferNN1D(config.model)
solver = IncidentBoundaryPINN1D(config.solver)

# Train
trainer = PEPINNTrainer(config)
history, sigma = trainer.train(model, solver)
print(f"Final σ: {sigma:.6f}")
