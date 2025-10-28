import pandas as pd

from sklearn.model_selection import train_test_split

# FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. 
# To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future 
# behavior, set `pd.set_option('future.no_silent_downcasting', True)
pd.set_option('future.no_silent_downcasting', True)


class Loader:
    def __init__(self):
        self.database = None

        self.species_to_numbers = {
            "iris-setosa": 1,
            "iris-versicolor": 2,
            "iris-virginica": 3
        }

        self.numbers_to_species = {
            1: "iris-setosa",
            2: "iris-versicolor",
            3: "iris-virginica"
        }

    def load_data(self, file_path: str): # --------------------- Função que carrega os dados do arquivo CSV
        try:
            self.data = pd.read_csv(file_path)
            return self.data
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo {file_path} não encontrado.")
        
        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo: {e}")
        
    def split_data(self,features,target): # ------------ Função que sepra o dataset em treinamento e testes

        x_train, x_test, y_train, y_test = train_test_split(
            features, # input
            target, # output
            test_size=0.2, # separa 20% para o treinamento
            shuffle=True, # mantém aleatório a divisão
            random_state=42,
            stratify=target
        )

        return x_train,x_test,y_train,y_test
    
    def get_features_and_target(self,database_processed): # --------------- Função que separa os dados em feature (que serão o
                                       # --------------- input) e target (o output/objetivo)
        feature_columns = [
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm"
        ]

        target_column = "species"

        missing_columns = [column for column in feature_columns + [target_column] if column not in database_processed.columns]
        if missing_columns:
            raise KeyError(f"As seguintes colunas não foram encontradas no DataFrame: {missing_columns}")
        
        features = database_processed[feature_columns]
        target = database_processed[target_column]

        return features,target
    
    def convert_species_to_numbers(self,database): # ------- Função que converte as espécies de STRING para um valor
                                          # ------- inteiro
        database_copy = database.copy()
        database_copy["species"] = database_copy["species"].str.strip().str.lower()

        database_copy["species"] = (
            database_copy["species"]
            .replace({
                "iris-setosa": 1,
                "iris-versicolor": 2,
                "iris-virginica": 3,
                "iris setosa": 1,
                "iris versicolor": 2,
                "iris virginica": 3
            }).infer_objects(copy=False)
        )

        return database_copy
    
    def convert_number_to_species(self,number): # ----- Funçãoque converte um valor inteiro para a STRING
                                                  # ----- associada a ele
        return self.numbers_to_species.get(number,'Unknown')
        
    