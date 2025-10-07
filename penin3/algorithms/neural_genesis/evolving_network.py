class DynamicNeuron(nn.Module):
    def __init__(self, neuron_id: str, input_size: int):
        super().__init__()
        self.neuron_id, self.input_size = neuron_id, input_size
        self.weights = nn.Parameter(torch.randn(input_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU()
        self.fitness = 0.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = stable_shape(x, self.weights.shape[0])
        if x.dim() == 1: x = x.view(1, -1)
        signal = (x * self.weights).sum(dim=-1, keepdim=True) + self.bias
        return self.activation(signal)

class EvolvingNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int = 10, output_dim: int = 10):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.neurons = nn.ModuleDict()
        self.connections: Dict[str, list] = {}
        self.input_layer = nn.Linear(input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.evolution_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        if not self.neurons: 
            return self.output_layer(x)
            
        # Process through evolved neurons with improved connectivity
        neuron_outputs = {}
        for nid, neuron in self.neurons.items():
            # Get inputs for this neuron
            inputs = []
            for src in self.connections.get(nid, ['input']):
                if src == 'input':
                    inputs.append(x)
                elif src in neuron_outputs:
                    inputs.append(neuron_outputs[src])
                else:
                    inputs.append(torch.zeros_like(x))
            
            if inputs:
                combined_input = torch.cat(inputs, dim=-1)
                neuron_outputs[nid] = neuron(combined_input)
        
        if neuron_outputs:
            # Improved aggregation of neuron outputs
            stacked = torch.stack(list(neuron_outputs.values()), dim=0)
            # Use attention-like mechanism for better integration
            weights = torch.softmax(torch.randn(len(neuron_outputs)), dim=0)
            weighted_output = torch.sum(stacked * weights.unsqueeze(-1), dim=0)
            final_input = stable_shape(weighted_output, self.input_dim)
            return self.output_layer(final_input)
        
        return self.output_layer(x)
    
    def evolve_architecture(self, performance_feedback: Dict = None):
        """Evolve network architecture based on performance feedback"""
        if performance_feedback is None:
            performance_feedback = {}
        
        # Add new neurons based on complexity needs
        current_complexity = len(self.neurons)
        target_complexity = performance_feedback.get('target_complexity', current_complexity + 1)
        
        if current_complexity < target_complexity:
            # Add new neuron
            new_id = f"evolved_{len(self.neurons)}"
            input_size = self.input_dim + len(self.neurons)  # Can connect to existing neurons
            new_neuron = DynamicNeuron(new_id, input_size)
            self.neurons[new_id] = new_neuron
            
            # Create intelligent connections
            if len(self.neurons) > 1:
                # Connect to best performing neurons
                existing_neurons = list(self.neurons.keys())[:-1]  # Exclude the new one
                if existing_neurons:
                    # Connect to random subset of existing neurons
                    num_connections = min(3, len(existing_neurons))
                    connections = random.sample(existing_neurons, num_connections)
                    connections.append('input')  # Always connect to input
                    self.connections[new_id] = connections
            else:
                self.connections[new_id] = ['input']
        
        # Record evolution event
        self.evolution_history.append({
            'timestamp': time.time(),
            'neurons_count': len(self.neurons),
            'connections_count': sum(len(conns) for conns in self.connections.values()),
            'performance_feedback': performance_feedback
        })

class IA3Brain:
    def __init__(self, input_dim=10, output_dim=10):
        self.brain = EvolvingNeuralNetwork(input_dim=input_dim, output_dim=output_dim)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.training_history = []
        
    def evolve_cycle(self, train_steps=10, performance_data=None):
        """Enhanced evolution cycle with actual training and architecture evolution"""
        # Generate training data based on current architecture
        batch_size = 32
        input_data = torch.randn(batch_size, self.brain.input_dim)
        
        # Create targets based on performance feedback
        if performance_data and 'target_actions' in performance_data:
            targets = torch.tensor(performance_data['target_actions'])
            if targets.shape[0] != batch_size:
                targets = targets.repeat(batch_size // targets.shape[0] + 1)[:batch_size]
        else:
            # Generate synthetic targets for training
            targets = torch.randn(batch_size, self.brain.output_dim)
        
        # Training loop
        total_loss = 0.0
        for step in range(train_steps):
            self.optimizer.zero_grad()
            outputs = self.brain(input_data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / train_steps
        
        # Evolve architecture if performance suggests it
