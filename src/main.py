from Evaluator import Evaluator

def main():
    evaluator = Evaluator()

    try:
        # carregando e preparando os dados
        print("================ Carregando o dataset! ================")
        evaluator.preparing_data("data/iris.csv")

        # treinando o modelo
        print("================ Treinando o classificador! ================")
        evaluator.train_model()

        # testando o modelo
        print("================ Testando o modelo! ================")
        accuracy, f1 = evaluator.evaluate_model()
        with open("resultados.txt", "w") as f:
            f.write("================ Resultados da predição! ================\n")
            f.write(f"Acurácia: {accuracy:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        print("Métricas salvas em 'resultados.txt'!")

        print("================================================================")
        print("================ Entrada de dados do usuário ================")
        print("================================================================")

        while True:
            try:
                print("Se deseja sair digite 'sair'")
                print("Comprimento da sepala em cm: ")
                sepal_length = input()
                if sepal_length == 'sair':
                    break
                sepal_length = float(sepal_length)
                sepal_width = float(input("Largura da sepala em cm: "))
                petal_length = float(input("Comprimento da pétala em cm: "))
                petal_width = float(input("Largura da pétala em cm: "))

                result = evaluator.predict_with_custom_data(sepal_length, sepal_width, petal_length, petal_width)
                print(f"Essa íris pertence à espécie **{result}**")
            except Exception as e:
                print("Erro na predição: {e}")
            except ValueError:
                print("Por favor entre com valores válidos!")
    except FileNotFoundError:
        print("Por favor, verifique se o arquivo 'data/iris.csv' existe e se está na pasta correta")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()