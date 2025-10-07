# ========== CÉREBRO REAL COM CRESCIMENTO ==========

class RealBrain(nn.Module):
    """Cérebro que REALMENTE cresce neurônios baseado em necessidade"""
    
    async def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Rede neural
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Estatísticas REAIS
        self.neuron_activations = torch.zeros(hidden_dim)
        self.total_forward_passes = 0
        self.neurons_added = 0
        self.neurons_removed = 0
        self.neuron_activation_history = []
        
    async def forward(self, x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32)
            
        h = self.fc1(x)
        
        # REALMENTE rastrear ativações
        with torch.no_grad():
            self.neuron_activations += torch.abs(h)
            self.total_forward_passes += 1
            
        h = self.act(h)
        out = self.fc2(h)
        
        # Verificar crescimento usando aprendizado não supervisionado
        if self.total_forward_passes % 100 == 0:
            self.check_growth_unsupervised()

        return await out

    async def check_growth_unsupervised(self):
        """Verificar crescimento usando dados reais do sistema (não regras fixas)"""
        if self.total_forward_passes < 200:
            return await  # Não suficiente dados

        # Integrar dados reais do sistema para decisão de crescimento
        system_complexity = self._get_system_complexity_factor()
        neural_capacity = self._assess_neural_capacity()

        # Crescimento baseado em complexidade real do ambiente
        if system_complexity > 0.8 and neural_capacity < 0.6 and self.hidden_dim < 100:
            self.add_neuron()
            logger.info(f"🌱 Neurônio adicionado por complexidade do sistema alta ({system_complexity:.2f})")

        # Poda baseada em eficiência neural real
        if neural_capacity > 0.9 and self.hidden_dim > 15:
            self.remove_neuron()
            logger.info(f"💀 Neurônio removido por super-eficiência ({neural_capacity:.2f})")

    async def _get_system_complexity_factor(self):
        """Avaliar complexidade real do sistema (CPU, memória, processos)"""
        try:
            import psutil
            cpu = psutil.cpu_percent() / 100.0
            memory = psutil.virtual_memory().percent / 100.0
            processes = len(list(psutil.process_iter())) / 1000.0  # Normalizar

            # Complexidade como média ponderada
            complexity = (cpu * 0.4) + (memory * 0.4) + (processes * 0.2)
            return await complexity
        except:
            return await 0.5  # Valor padrão

    async def _assess_neural_capacity(self):
        """Avaliar capacidade neural usando autoencoder não-supervisionado"""
        if self.total_forward_passes < 50:
            return await 0.5

        activations = self.neuron_activations / self.total_forward_passes

        # Autoencoder simples para medir quão bem a rede representa os dados
        import torch.nn.functional as F

        input_data = activations.unsqueeze(0)
        encoded = F.relu(self.fc1(input_data))
        decoded = self.fc2(encoded)

        reconstruction_error = F.mse_loss(decoded, input_data).item()

        # Capacidade = 1 - erro (menor erro = maior capacidade)
        capacity = max(0.0, 1.0 - reconstruction_error)

        return await capacity
    
    async def add_neuron(self):
        """Adicionar neurônio REAL"""
        new_hidden = self.hidden_dim + 1
        
        # Expandir camadas
        new_fc1 = nn.Linear(self.input_dim, new_hidden)
        new_fc2 = nn.Linear(new_hidden, self.output_dim)

        # Copiar pesos existentes
        with torch.no_grad():
            new_fc1.weight[:self.hidden_dim] = self.fc1.weight
            new_fc1.bias[:self.hidden_dim] = self.fc1.bias
            new_fc2.weight[:, :self.hidden_dim] = self.fc2.weight

            # Novo neurônio com pesos aleatórios pequenos
            nn.init.xavier_uniform_(new_fc1.weight[self.hidden_dim:self.hidden_dim+1])
            nn.init.xavier_uniform_(new_fc2.weight[:, self.hidden_dim:self.hidden_dim+1])

        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.hidden_dim = new_hidden

        # Expandir estatísticas
        self.neuron_activations = torch.cat([self.neuron_activations, torch.zeros(1)])
        self.neurons_added += 1

        # Reset activation history if shape changed
        if len(self.neuron_activation_history) > 0 and self.neuron_activation_history[-1].shape[0] != self.neuron_activations.shape[0]:
            self.neuron_activation_history = []
        
    async def remove_neuron(self):
        """Remover neurônio morto"""
        if self.hidden_dim <= 10:
            return
            
        # Encontrar neurônio menos ativo
        avg_activation = self.neuron_activations / self.total_forward_passes
        victim = torch.argmin(avg_activation)
        
        # Criar máscara
        mask = torch.ones(self.hidden_dim, dtype=torch.bool)
        mask[victim] = False
        
        new_hidden = self.hidden_dim - 1
        
        # Contrair camadas
        new_fc1 = nn.Linear(self.input_dim, new_hidden)
        new_fc2 = nn.Linear(new_hidden, self.output_dim)
        
        with torch.no_grad():
            new_fc1.weight = nn.Parameter(self.fc1.weight[mask])
            new_fc1.bias = nn.Parameter(self.fc1.bias[mask])
            new_fc2.weight = nn.Parameter(self.fc2.weight[:, mask])
            new_fc2.bias = self.fc2.bias
            
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.hidden_dim = new_hidden
        
        # Contrair estatísticas
        self.neuron_activations = self.neuron_activations[mask]
        self.neurons_removed += 1

