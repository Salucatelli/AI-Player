local estadoFile = "game_state.csv"
local acaoFile = "action.csv"

-- Função para verificar se estamos no mapa (overworld)
function esta_no_mapa()
    local tela = mainmemory.readbyte(0x0100)
    return tela == 0
end

-- Função para entrar na primeira fase
function entrar_na_primeira_fase()
    local input = {}
    input["Right"] = true
    joypad.set(input)
    emu.frameadvance()
    input = {}
    input["A"] = true
    joypad.set(input)
    emu.frameadvance()
    print("Entrando na primeira fase...")
end

-- Função para salvar o estado atual
function salvar_estado()
    local file = io.open(estadoFile, "w")
    if file then
        local marioX = mainmemory.read_u16_le(0x0094)
        local marioY = mainmemory.read_u16_le(0x0096)
        local score  = mainmemory.read_u24_le(0x0F34)
        local vidas  = mainmemory.readbyte(0x0DBE)
        local morto  = mainmemory.readbyte(0x0071)

        file:write("marioX,marioY,score,vidas,morto\n")
        file:write(string.format("%d,%d,%d,%d,%d\n", marioX, marioY, score, vidas, morto))
        file:close()
    end
end

-- Lê ação do Python
function ler_acao()
    local file = io.open(acaoFile, "r")
    if not file then return nil end
    local acao = file:read("*l")

    -- if acao and acao ~= "" then
    --     print("🔹 Ação recebida:", acao)
    --     executar_acao(acao)
    -- else
    --     print("⚠️ Nenhuma ação no arquivo.")
    -- end

    file:close()
    return acao
end

-- Executa a ação
function executar_acao(acao)
    if acao == nil then
        return
    end

    local input = {
        ["Left"]  = false,
        ["Right"] = false,
        ["A"]     = false,
        ["B"]     = false
    }

    local comando = ""

    if acao == "left" then
        comando = comando .. "Left"
    elseif acao == "right" then
        comando = comando .. "Right"
    elseif acao == "jump" then
        comando = comando .. "B"   -- pular = botão B no SNES
    elseif acao == "run" then
        comando = comando .. "Y"   -- correr = botão Y no SNES
    else
        comando = comando .. "|"
    end

    -- Define o input no frame atual
    --joypad.set({Right = true})

    if acao == "left" then
        joypad.set({Left = true}, 1)
    elseif acao == "right" then
        joypad.set({Right = true}, 1)
    elseif acao == "jump" then
        joypad.set({A = true}, 1)
    elseif acao == "run" then
        joypad.set({B = true}, 1)
    end

    -- -- Usa controle 1 (número, não string)
    -- joypad.set(input)
end

-- Reinicia o jogo se o Mario morrer
function reiniciar_se_morreu()
    local morto = mainmemory.readbyte(0x0071)
    if morto == 9 then
        print("Mario morreu! Reiniciando...")
        savestate.loadslot(1)  -- carrega save state inicial
    end
end

-- Loop principal
while true do
    if esta_no_mapa() then
        entrar_na_primeira_fase()
    else
        salvar_estado()
        local acao = ler_acao()

        -- === Isso executa a ação
        if acao then
            executar_acao(acao)
        end
        
        reiniciar_se_morreu()
    end
    emu.frameadvance()
end
