using Flux
using Flux.Tracker: update!
using Flux.Tracker
using Random
using LinearAlgebra

Random.seed!(13)

# Training data
X = [0 0; 0 1; 1 0; 1 1]
Y = [-1.0, 1, 1, -1.0]

# First layer: form input to hidden layer
# 2 input states, 3 hidden states
α = param(rand(3, 2))
α₀ = param(0.3*rand(3))

# Second layer: form hidden to output layer
# 3 hidden states, 2 output states 
β = param(rand(1, 3))
β₀ = param(0.3*rand(1))


# Model of Neural Network: two layers
f₁(x) = α * x .+ α₀
f₂(x) = β * x .+ β₀

# Multilayer NN: Composite function
f(x) = f₂(σ.(f₁(x)))

# Loss function: square error
L(x, y) = sum((y .- f(x)).^2)

# Gradient of f, i-th input of training data
∇L(i) = Tracker.gradient(() -> L(X[i,:], Y[i]), params(α, α₀, β, β₀))

# Training loop
for it in 1:2000
	# Loop over the Training set
	Δ₁, Δ₂, Δ₃, Δ₄ = 0.0, 0.0, 0.0, 0.0
	for i=1:4
		Δ₁ = Δ₁ .+ ∇L(i)[α]
		Δ₂ = Δ₂ .+ ∇L(i)[α₀]
		Δ₃ = Δ₃ .+ ∇L(i)[β]
		Δ₄ = Δ₄ .+ ∇L(i)[β₀]
	end
	# Update model weights
	update!(α, -0.1Δ₁)
	update!(α₀, -0.1Δ₂)
	update!(β, -0.1Δ₃)
	update!(β₀, -0.1Δ₄)
	
	# Compute Error
	E = sum([L(X[i,:], Y[i]) for i=1:4])
	# Logging and stopping condition
	println(it, " --> Loss: ", E, " Weights: ", norm([norm(α), norm(α₀), norm(β), norm(β₀)]))
	if E < 0.01
		break
	end
end

println([f(X[i,:]) for i=1:4])