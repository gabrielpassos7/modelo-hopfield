import funcoes as f

print("----- TESTE DA REDE HOPFIELD -----\n")

X, y = f.carregar_amostras("arquivos_texto/Dados.xlsx")

pesos = f.calcular_pesos(X)

print("Padrões e pesos carregados!")

print("\nIniciando reconhecimento...\n")

estado_final, epocas = f.reconhecer_padrao(y, pesos)

indice, erro_final = f.encontrar_padrao_associado(estado_final, X)

erro_entrada = f.encontrar_padrao_associado(y, X)[1]

print("\n----- RESULTADO FINAL -----")
print(f"Padrão associado: {indice + 1}")
print(f"Épocas necessárias: {epocas}")
print(f"Erro da entrada original: {erro_entrada} elementos")
print(f"Erro após a recuperação: {erro_final} elementos")
print(f"Erro percentual final: {(erro_final / len(estado_final)) * 100:.2f}%")