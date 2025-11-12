import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os

# Caminhos
BASE_DIR = r"D:\Prog\Faculdade\8º período\Inteligencia Artificial\IA-player-de-games\lua_scripts"
STATE_FILE = os.path.join(BASE_DIR, "game_state.csv")
ACTION_FILE = os.path.join(BASE_DIR, "action.csv")

# Modelo simples de rede neural
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Cria modelo e otimizador
model = SimpleNN(input_size=4, output_size=4)  # 4 estados, 4 ações
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lista de ações
actions = ["run", "left", "right", "jump"]

# Loop principal
last_line = 0
print("🚀 IA iniciada. Aguardando estados do jogo...")

while True:
    if not os.path.exists(STATE_FILE):
        time.sleep(0.1)
        print("Aqruivo não encontrado")
        continue

    try:
        data = pd.read_csv(STATE_FILE)
        if len(data) <= last_line:
            time.sleep(0.05)
    except Exception as e:
        print(e)
        continue
    # data = pd.read_csv(STATE_FILE)
    # if len(data) <= last_line:
    #     time.sleep(0.05)
    #     continue

    # Pega último estado
    state = data.iloc[-1, 1:].values  # ignora coluna 'frame'
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Gera previsão da rede
    with torch.no_grad():
        output = model(state_tensor)
        action_index = torch.argmax(output).item()
        action = actions[action_index]

    # Escreve ação no CSV
    with open(ACTION_FILE, "w") as f:
        f.write(action)

    last_line = len(data)
    time.sleep(0.05)  # pequeno delay para sincronizar frames
