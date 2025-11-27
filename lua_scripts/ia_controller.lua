local estadoFile = "game_state.csv"
local acaoFile = "action.csv"

-- Variável para controlar se o cabeçalho já foi escrito
local header_written = false

-- Variável para controlar se iniciou a primeira fase
local iniciou_fase = false

-- Função para verificar se estamos no mapa (overworld)
function esta_no_mapa()
    local tela = mainmemory.readbyte(0x001A)
    --print("O valor da tela é " .. tela)
    return tela
end

-- Entra automaticamente na primeira fase
function entrar_na_primeira_fase()
    print("Movendo Mario até a primeira fase...")

    -- Endereço da posição do Mario no mapa (Overworld)
    local pos_anterior = mainmemory.readbyte(0x1F17)
    local frames_sem_mover = 0

    -- Segura para a direita até detectar movimento no mapa
    while true do
        joypad.set({Right = true}, 1)
        emu.frameadvance()

        local pos_atual = mainmemory.readbyte(0x1F17)
        if pos_atual ~= pos_anterior then
            frames_sem_mover = 0
            pos_anterior = pos_atual
        else
            frames_sem_mover = frames_sem_mover + 1
        end

        -- Se ficou parado por muitos frames, deve ter chegado na fase
        if frames_sem_mover > 40 then
            print("Pressionando A para entrar na fase...")
            for i = 1, 40 do
                joypad.set({A = true}, 1)
                emu.frameadvance()
            end
            break
        end
    end

    iniciou_fase = true
end

-- Função para limpar o arquivo CSV
local function limpar_arquivo_estado()
    local file = io.open(estadoFile, "w") -- "w" recria o arquivo vazio
    if file then
        print("Arquivo game_state.csv limpo após morte do Mario.")
    else
        print("Erro ao limpar o arquivo game_state.csv.")
    end
end

-- Função para salvar o estado atual do jogo
function salvar_estado()
    local file = io.open(estadoFile, "a") -- Abre em modo 'append' (anexar)
    if file then
        local marioX = mainmemory.read_u16_le(0x0094)
        local marioY = mainmemory.read_u16_le(0x0096)
        local score  = mainmemory.read_u24_le(0x0F34)
        local vidas  = mainmemory.readbyte(0x0DBE)
        local morto  = mainmemory.readbyte(0x0071)

        -- Escreve o cabeçalho apenas uma vez
        if not header_written then
            file:write("marioX,marioY,score,vidas,morto\n")
            header_written = true
        end
        
        -- Escreve o estado do jogo
        file:write(string.format("%d,%d,%d,%d,%d\n", marioX, marioY, score, vidas, morto))
        file:close()
    end
end

-- Lê ação do Python
function ler_acao()
    local file = io.open(acaoFile, "r")
    if not file then return nil end
    local acao = file:read("*l")

    file:close()
    return acao
end

-- Executa a ação
function executar_acao(acao)
    if acao == nil then
        return
    end

    -- Limpa todos os inputs antes de aplicar o novo
    joypad.set({}) 

    -- Ações devem ser minúsculas para corresponder ao Python
    if acao == "left" then
        joypad.set({Left = true}, 1)
    elseif acao == "right" then
        joypad.set({Right = true}, 1)
    elseif acao == "jump" then
        joypad.set({A = true}, 1) -- pular = botão A no SNES (comum)
    elseif acao == "run" then
        joypad.set({B = true}, 1) -- correr = botão B no SNES (comum)
    end
    
    -- Adicionando suporte para ações combinadas (ex: "right+jump")
    -- local actions = {}
    -- for action in acao:gmatch("([^%+%s]+)") do
    --     if action == "left" then
    --         actions.Left = true
    --     elseif action == "right" then
    --         actions.Right = true
    --     elseif action == "jump" then
    --         actions.A = true
    --     elseif action == "run" then
    --         actions.B = true
    --     end
    -- end
    
    -- if next(actions) ~= nil then
    --     joypad.set(actions, 1)
    -- end
end

-- Reinicia o jogo se o Mario morrer
function reiniciar_se_morreu()
    local morto = mainmemory.readbyte(0x0071)
    if morto == 9 then
        print("Mario morreu! Reiniciando...")
        
        -- carrega save state inicial
        savestate.loadslot(1)  

        -- Aguarda mais 1 frame para estabilizar
        emu.frameadvance()

        -- Limpa o arquivo de estado
        limpar_arquivo_estado() 

        iniciou_fase = false

        -- Reseta o flag do cabeçalho para reescrever no próximo episódio
        header_written = false
    end
end

-- Loop principal
while true do
    local current_place = esta_no_mapa()

    --print(current_place)

    if(current_place == 0) then
        iniciou_fase = true
    end

    -- print(current_place)
    if iniciou_fase then
        salvar_estado()
        local acao = ler_acao()

        -- === Isso executa a ação
        if acao then
            executar_acao(acao)
        end
        
        reiniciar_se_morreu()
    end
    -- if not iniciou_fase and current_place == 239 then
    --     print("Esta no mapa")
    --     entrar_na_primeira_fase()
    -- elseif iniciou_fase then
    --     salvar_estado()
    --     local acao = ler_acao()

    --     -- === Isso executa a ação
    --     if acao then
    --         executar_acao(acao)
    --     end
        
    --     reiniciar_se_morreu()
    -- end
    emu.frameadvance()
end
