from src.Loader import Loader

def main():
    
    print("============ SISTEMA DE CLASSIFICAÇÃO DE ESPÉCIES DE ÍRIS ============")

    # ----------------------------------------------------------------------------- Etapas para realização:
    # ----------------------------------------------------------------------------------- LEITURA DOS DADOS
    
    loader = Loader()
    database = loader.load_data("data/iris.csv")
    
    #-- print(data) # testando se está puxando os dados corretamente
    
    # ----------------------------------------------------------------------------- PROCESSAMENTO DOS DADOS
    database_processed = loader.convert_species_to_numbers(database)
    features, target = loader.get_features_and_target(database_processed)

    # ----------------------------------------------------------------------------------- DIVISÃO DOS DADOS
    x_train,x_test,y_train,y_test = loader.split_data(features,target)
    
    #-- print(f"Treino: {len(x_train)} amostras") #testando se o tamanho de treino esta certo
    #-- print(f"Teste: {len(x_test)} amostras") # testando se o tamanho de teste está certo


if __name__ == "__main__":
    main()