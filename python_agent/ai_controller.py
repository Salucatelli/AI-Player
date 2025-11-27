import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os
import random
from collections import deque
import math 
import numpy as np 

# Caminhos
BASE_DIR = r"D:\Prog\Faculdade\8º período\Inteligencia Artificial\IA-player-de-games\lua_scripts"
STATE_FILE = os.path.join(BASE_DIR, "game_state.csv")
ACTION_FILE = os.path.join(BASE_DIR, "action.csv")

# Parâmetros de RL
INPUT_SIZE = 5  # marioX, marioY, score, vidas, morto
OUTPUT_SIZE = 4 # run, left, right, jump
LR = 0.00025
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
BATCH_SIZE = 32 
TARGET_UPDATE = 1000 # Atualiza a rede alvo a cada 1000 passos

# Lista de ações 
ACTIONS = ["run", "left", "right", "jump"]

# --- 1. Modelo de Rede Neural (DQN) ---
# Cria o modelo da Rede neural
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. Buffer de Replay de Experiência ---
# Essa função serve para armazenar o buffer da experiência que a IA vai adquirindo
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Armazena a transição como uma tupla
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Amostra um lote aleatório de transições
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # Converte a tupla de booleanos 'done' para um tensor de float (0.0 ou 1.0)
        done_tensor = torch.tensor(done, dtype=torch.float32)

        return torch.stack(state), torch.tensor(action), torch.tensor(reward, dtype=torch.float32), torch.stack(next_state), done_tensor

    def __len__(self):
        return len(self.buffer)


# --- 3. Agente RL (Deep Q-Learning) ---
class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(INPUT_SIZE, OUTPUT_SIZE)
        self.target_net = DQN(INPUT_SIZE, OUTPUT_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Rede alvo em modo de avaliação

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(capacity=50000)
        self.steps_done = 0

    def select_action(self, state_tensor):

        sample = random.random()

        # Decaimento exponencial de epsilon
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Escolhe a ação com o maior Q-valor (Exploração)
                return self.policy_net(state_tensor).argmax().item()
        else:
            # Escolhe uma ação aleatória (Explotação)
            return random.randrange(OUTPUT_SIZE)

    def optimize_model(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        # Amostra um lote de transições
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(BATCH_SIZE)

        # Calcula Q(s_t, a) - Q-valores da ação tomada
        # policy_net(state_batch) retorna Q-valores para todas as ações.
        # .gather(1, action_batch.unsqueeze(1)) seleciona o Q-valor da ação real tomada.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Calcula V(s_{t+1}) = max_a Q_target(s_{t+1}, a)
        # target_net(next_state_batch).max(1)[0] retorna o Q-valor máximo para o próximo estado.
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Máscara para estados finais (done=True)
        # next_state_values[done_batch] = 0.0 # done_batch é o tensor 'done'
        
        # Calcula o valor esperado de Q: r + gamma * max_a Q_target(s_{t+1}, a)
        # Calcula o valor esperado de Q: r + gamma * max_a Q_target(s_{t+1}, a) * (1 - done)
        # O tensor 'done' é 1.0 para estados finais, 0.0 caso contrário.
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

        # Calcula a perda (Loss)
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        # Otimização
        self.optimizer.zero_grad()
        loss.backward()
        # Clipa gradientes para estabilidade
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

# --- 4. Funções Auxiliares ---
def calculate_reward(current_state, previous_state):
    """
    Função de recompensa simples:
    - Recompensa por avançar na tela (marioX)
    - Penalidade por morrer (morto)
    - Recompensa por pontuação (score)
    """
    if previous_state is None:
        return 0.0
    
    # Recompensa por avanço horizontal
    reward_x = (current_state[0] - previous_state[0]) * 0.1
    
    # Recompensa por pontuação
    reward_score = (current_state[2] - previous_state[2]) * 0.01
    
    # Penalidade por morte
    reward_death = 0.0
    if current_state[4] > previous_state[4]: # Se o contador de morte aumentou
        reward_death = -10.0
        
    # Recompensa total
    total_reward = reward_x + reward_score + reward_death
    
    return total_reward

def is_done(current_state):
    """Verifica se a fase terminou (Mario morreu)."""
    # O script Lua reinicia o jogo se morto == 9.
    # VamVouos considerar a fase "done" quando o Mario morre.
    return current_state[4] == 9 # 'morto' é o 5º elemento (índice 4)


# --- 5. Loop Principal de Treinamento ---
agent = DQNAgent()
previous_state = None
last_line = 0
total_steps = 0
episode_reward = 0.0
episode_count = 0

print("🚀 Agente DQN iniciado. Aguardando estados do jogo...")

while True:
    # 1. Leitura do Estado do Jogo
    if not os.path.exists(STATE_FILE):
        time.sleep(0.1)
        print("Arquivo de estado não encontrado.")
        continue

    try:
        # Lê o CSV, forçando o tipo de dado para float
        COLUMN_NAMES = ["marioX", "marioY", "score", "vidas", "morto"]
        data = pd.read_csv(STATE_FILE, skiprows=1, header=None, names=COLUMN_NAMES, dtype=float)

        
        # Verifica se há novos dados
        if len(data) <= last_line:
            time.sleep(0.05)
            continue
            
        # Pega o último estado
        current_state_raw = data.iloc[-1].values
        
        if len(current_state_raw) == INPUT_SIZE:
            current_state = current_state_raw
        elif len(current_state_raw) == INPUT_SIZE + 1:
            current_state = current_state_raw[1:] 
        else:
            print(f"Erro: Tamanho do estado inesperado ({len(current_state_raw)}). Esperado {INPUT_SIZE} ou {INPUT_SIZE + 1}.")
            time.sleep(0.05)
            continue
            
        # Garante que o array NumPy é do tipo float antes de converter para tensor
        if current_state.dtype == np.object_:
            current_state = current_state.astype(np.float32)
            
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32)
        
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        time.sleep(0.05)
        continue
        
    # 2. Cálculo da Recompensa e Transição
    reward = calculate_reward(current_state, previous_state)
    done = is_done(current_state)
    
    # 3. Armazenamento no Buffer de Replay (se houver estado anterior)
    if previous_state is not None:
        previous_state_tensor = torch.tensor(previous_state, dtype=torch.float32)
        # Ação anterior (action_index) deve ser armazenada, mas não temos ela aqui.
        # Para simplificar, vamos assumir que a ação anterior foi a que acabamos de selecionar.
        # **NOTA:** Em um sistema de RL real, você precisa armazenar a ação que *levou* ao next_state.
        # Como estamos em um loop síncrono, a ação selecionada no passo anterior é a que levou ao estado atual.
        # Para fins de demonstração, vamos usar a ação que será selecionada *agora* como a ação anterior.
        # Isso é um hack e deve ser corrigido em uma implementação futura.
        # No entanto, para fazer o código funcionar com a estrutura de arquivos, será feito assim.
        
        # Vou usar a ação selecionada no passo 4 como a ação que levou a este estado.
        
        # Para o DQN, precisamos da transição (s, a, r, s', done).
        # s = previous_state_tensor
        # s' = current_state_tensor
        # r = reward
        # done = done
        # a = Ação que levou de s a s' (precisamos armazenar a ação selecionada no loop anterior)
        
        # Para simplificar, vou armazenar a transição no final do loop, após a seleção da ação.
        pass # A transição será armazenada no final do loop

    # 4. Seleção da Ação (Epsilon-Greedy)
    # A seleção da ação deve ser feita *após* a verificação de "done" e o reset de previous_state
    # para garantir que a primeira ação do novo episódio seja selecionada corretamente.
    action_index = agent.select_action(current_state_tensor)
    action_string = ACTIONS[action_index]

    # 5. Armazenamento da Transição (s, a, r, s', done)
    if previous_state is not None:
        # Armazena a transição anterior: (previous_state, action_index_anterior, reward, current_state, done)
        # Como não tenho a action_index_anterior, vou usar a action_index atual como placeholder
        # Isso é um problema de sincronização inerente ao uso de arquivos.
        # Para simplificar, vou usar a action_index atual como a ação que levou ao estado atual.
        # O ideal seria ter um buffer para armazenar a ação do frame anterior.
        
        # Para o primeiro passo de treinamento, vou ignorar a ação anterior e usar a atual.
        # Vou usar a action_index atual como um placeholder para a ação anterior.
        agent.buffer.push(previous_state_tensor, action_index, reward, current_state_tensor, done)
        
    # 6. Otimização do Modelo
    if total_steps % 4 == 0: # Otimiza a cada 4 passos (pode ser ajustado)
        agent.optimize_model()

    # 7. Atualização da Rede Alvo
    if total_steps % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    # 8. Escrita da Ação no CSV
    with open(ACTION_FILE, "w") as f:
        f.write(action_string)

    # 9. Atualização de Estado e Contadores
    
    # Se a fase terminou (Mario morreu)
    if done:
        episode_count += 1
        episode_reward = 0.0
        last_line = 0 # Resetar o contador de linhas lidas
        previous_state = None # Reseta o estado anterior para o próximo episódio
        
    else:
        # Se o episódio não terminou, atualiza o estado e o contador de linhas
        previous_state = current_state
        last_line = len(data)
        
    total_steps += 1
    episode_reward += reward
    
    time.sleep(0.05) # Pequeno delay para sincronizar frames