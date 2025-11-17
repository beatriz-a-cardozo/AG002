from Loader import Loader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score

class Evaluator:
    def __init__(self):
        self.loader = Loader()
        self.model= None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def preparing_data(self, file_path):
        data = self.loader.load_data(file_path) #carregando os dados

        processed_data = self.loader.convert_species_to_numbers(data) # converte os dados carregados de especies para numeros

        features, target = self.loader.get_features_and_target(processed_data) # obtem as features/target

        #separação de treino e teste usando as features e target obtidas acima /\
        self.x_train, self.x_test, self.y_train, self.y_test = self.loader.split_data(features, target)

        return True

    def train_model(self):

        # Escolhemos o KNN das opções disponíveis

        self.model = KNeighborsClassifier() # usando o knn do sci kit learn

        # Realizando o treinamento do classificador
        self.model.fit(self.x_train, self.y_train)
    
    def evaluate_model(self):

        if self.model is None:
            raise Exception("O classificador não foi treinado")

        #realizando os testes
        y_pred = self.model.predict(self.x_test)


        # Salvando as métricas do teste
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="weighted")
        print("=========== METRICAS DA PREDIÇÃO ===========")
        print(f'Acurácia: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')

        #retornando com as métricas salvas
        return accuracy, f1
    
    def predict_with_custom_data(self, sepal_length, sepal_width, petal_length, petal_width):
        #checando primeiro se o modelo foi treinado
        if self.model is None:
            raise Exception("O classificador não foi treinado")
        
        custom_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        
        # Usando a entrada do usuario para avaliar que espécie ele inseriu
        prediction = self.model.predict(custom_data)
        species = self.loader.convert_number_to_species(prediction[0])

        return species

