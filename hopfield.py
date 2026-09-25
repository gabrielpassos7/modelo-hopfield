import funcoes as f

print("\nIniciando cálculo dos pesos...\n")

X, y = f.carregar_amostras("arquivos_texto/Dados.xlsx")

print(f"Padrões carregados: {len(X)}")
print(f"Neurônios (Elementos): {X.shape[1]}")

pesos = f.calcular_pesos(X)

print("\n----- RESULTADO DO CÁLCULO DOS PESOS -----")
print("\nMatriz de pesos:")
print(pesos)

print("\nCálculo concluído!")