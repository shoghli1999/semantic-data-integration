import os
import numpy as np
import pandas as pd
from thefuzz import fuzz
from scipy import spatial
from jarowinkler import *
import gensim.downloader as api
from gensim.models import KeyedVectors

def main():
    # C:\Users\shirin\gensim-data
    # choose from multiple models https://github.com/RaRe-Technologies/gensim-data  glove-wiki-gigaword-50
    print('loding word2vec ...')
    model = KeyedVectors.load_word2vec_format('C:\\Users\\shirin\\gensim-data\\glove-twitter-25\\glove-twitter-25.gz', binary=False)
    print('loding word2vec finished')

    columns_datasetOne = ['Name', 'Address', 'City', 'ZIPCode', 'County', 'Country', 'Website', 'Employee']
    columns_datasetTwo = ['University', 'Url', 'Destination', 'ZIP', 'Township', 'Working']
    columns_datasetThree = ['Uni', 'Location', 'Kingdom', 'Town', 'Post']
    columns_MediatedSchema = ['University', 'Address', 'City', 'ZIPCode', 'County', 'Country', 'Website', 'Employee']
    
    # Read all three Datasets
    datasetOne = pd.read_excel(os.path.dirname(__file__)+'\\CollegesUniversities.xlsx', usecols = columns_datasetOne, nrows = 2);
    datasetTwo = pd.read_excel(os.path.dirname(__file__)+'\\CollegeUniversityCampuses.xlsx', usecols = columns_datasetTwo, nrows = 2);
    datasetThree = pd.read_excel(os.path.dirname(__file__)+'\\NationalUniversitiesRankings.xlsx', usecols = columns_datasetThree, nrows = 2);

    # Concatenate all three Datasets
    dataFrames = [datasetOne, datasetTwo, datasetThree];
    concatResult = pd.concat(dataFrames)
    
    # Create Mediated Schema if not exist
    mediatedSchemaPath=os.path.dirname(__file__)+'\\MediatedSchema.xlsx';
    if(os.path.isfile(mediatedSchemaPath) == False):
        concatResult.to_excel(mediatedSchemaPath); 

    datasetMediatedSchema = pd.read_excel(mediatedSchemaPath, usecols = columns_MediatedSchema,nrows = 2);
    
############################################################# DATASET ONE #################################################################

    # Get all column attributes of three dataSets and Mediated Schema
    datasetOne_attributes = GetAttributeOfColumns(datasetOne)
    mediatedSchema_attributes = GetAttributeOfColumns(datasetMediatedSchema)

    # Create Matrix for each macher in datasetOne 
    matrix_minimum_combiner_datasetOne = np.zeros((len(datasetOne_attributes),len(mediatedSchema_attributes)))
    matrix_maximum_combiner_datasetOne = np.zeros((len(datasetOne_attributes),len(mediatedSchema_attributes)))
    matrix_average_combiner_datasetOne = np.zeros((len(datasetOne_attributes),len(mediatedSchema_attributes)))
    
    # Applying combination for Dataset One (Min, Max, Average Combiner)
    for idx, datasetOne_attribute in enumerate(datasetOne_attributes):
     for jdx, mediatedSchema_attribute in enumerate(mediatedSchema_attributes):
        editdistanceResult = round(fuzz.ratio(datasetOne_attribute,mediatedSchema_attribute)/100, 2)
        jarowinklerResult = round(jarowinkler_similarity(datasetOne_attribute, mediatedSchema_attribute), 2)
        word2VecResult = round(1 - spatial.distance.cosine(get_vector(datasetOne_attribute,model), 
                                                           get_vector(mediatedSchema_attribute,model)), 2)

        matrix_minimum_combiner_datasetOne[idx][jdx] = MinimumCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
        matrix_maximum_combiner_datasetOne[idx][jdx] = MaximumCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
        matrix_average_combiner_datasetOne[idx][jdx] = AverageCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
    
    similarities_matrix_datasetOne = [];
    similarities_matrix_datasetOne.append(matrix_minimum_combiner_datasetOne);
    similarities_matrix_datasetOne.append(matrix_maximum_combiner_datasetOne);
    similarities_matrix_datasetOne.append(matrix_average_combiner_datasetOne);

    # Create Ground Truth Matrix for datasetOne
    matrix_ground_truth_datasetOne = CreateGroundTruthMatrix(["0,0", "1,1", "2,2", "3,3", "4,4", "5,5", "6,6", "7,7"], 
                                                             len(datasetOne_attributes), len(mediatedSchema_attributes));

    for matrix in similarities_matrix_datasetOne:
     DoThreshold(0.5, matrix)
     cardinality_matrix = DoCardinalityDatasetOne(matrix, len(datasetOne_attributes), len(mediatedSchema_attributes))
     
     # Calculate True Positive
     TP = 0;
     positions_is_true_ground_truth_matrix = GetSpecificPositionsInMatrix(matrix_ground_truth_datasetOne, 1)
     for idx, x in enumerate(positions_is_true_ground_truth_matrix):
        a = int(positions_is_true_ground_truth_matrix[idx].split(',')[0])
        b = int(positions_is_true_ground_truth_matrix[idx].split(',')[1])
        if(matrix_ground_truth_datasetOne[a][b] == cardinality_matrix[a][b]):
         TP = TP + 1;
     
     # Calculate False Positive   داخل کامباینر کاردینالیتی نگاه کن کجا یک است که در ماترسی گراندتروس صفره
     # Positions that are one in Cardinality matrix
     # And are zero in the Ground Truth matrix
     FP = 0
     positions_is_true_cardinality_matrix = GetSpecificPositionsInMatrix(cardinality_matrix, 1)
     for idx, x in enumerate(positions_is_true_cardinality_matrix):
      a = int(positions_is_true_cardinality_matrix[idx].split(',')[0])
      b = int(positions_is_true_cardinality_matrix[idx].split(',')[1])
      if(matrix_ground_truth_datasetOne[a][b] == 0):
       FP = FP + 1;
     
     # Calculate False Negative  داخل گراندتروس یکه ولی در کامباینر کاردینالیتی صفره
     # Positions that are one in Ground Truth matrix
     # And are zero in the Cardinality matrix
     FN = 0
     for idx, x in enumerate(positions_is_true_ground_truth_matrix):
      a = int(positions_is_true_ground_truth_matrix[idx].split(',')[0])
      b = int(positions_is_true_ground_truth_matrix[idx].split(',')[1])
      if(cardinality_matrix[a][b] == 0):
       FN = FN + 1;
     
     precision = TP / (TP + FP)
     recall = TP / (TP + FN)
     f1 = 2 * (precision * recall) / (precision + recall)

     if(precision == 10):
        break;
    
############################################################# DATASET TWO #################################################################
    
    # Get all column attributes of dataSet two
    datasetTwo_attributes = GetAttributeOfColumns(datasetTwo)

    # Create Matrix for each macher in datasetTwo 
    matrix_minimum_combiner_datasetTwo = np.zeros((len(datasetTwo_attributes),len(mediatedSchema_attributes)))
    matrix_maximum_combiner_datasetTwo = np.zeros((len(datasetTwo_attributes),len(mediatedSchema_attributes)))
    matrix_average_combiner_datasetTwo = np.zeros((len(datasetTwo_attributes),len(mediatedSchema_attributes)))

    # Applying combination for Dataset Two (Min, Max, Average Combiner)
    for idx, datasetTwo_attribute in enumerate(datasetTwo_attributes):
     for jdx, mediatedSchema_attribute in enumerate(mediatedSchema_attributes):
        editdistanceResult = round(fuzz.ratio(datasetTwo_attribute,mediatedSchema_attribute)/100, 2)
        jarowinklerResult = round(jarowinkler_similarity(datasetTwo_attribute, mediatedSchema_attribute), 2)
        word2VecResult = round(1 - spatial.distance.cosine(get_vector(datasetTwo_attribute,model), 
                                                           get_vector(mediatedSchema_attribute,model)), 2)

        matrix_minimum_combiner_datasetTwo[idx][jdx] = MinimumCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
        matrix_maximum_combiner_datasetTwo[idx][jdx] = MaximumCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
        matrix_average_combiner_datasetTwo[idx][jdx] = AverageCombiner(editdistanceResult, jarowinklerResult, word2VecResult)

    similarities_matrix_datasetTwo = [];
    similarities_matrix_datasetTwo.append(matrix_minimum_combiner_datasetTwo);
    similarities_matrix_datasetTwo.append(matrix_maximum_combiner_datasetTwo);
    similarities_matrix_datasetTwo.append(matrix_average_combiner_datasetTwo);

    # Create Ground Truth Matrix for datasetTwo
    matrix_ground_truth_datasetTwo = CreateGroundTruthMatrix(["0,0", "1,6", "2,1", "3,3", "4,4", "5,7"], len(datasetTwo_attributes), 
                                                  len(mediatedSchema_attributes));

    for matrix in similarities_matrix_datasetTwo:
     DoThreshold(0.5, matrix)
     cardinality_matrix = DoCardinalityDatasetTwo(matrix, len(datasetTwo_attributes), len(mediatedSchema_attributes))
     
     # Calculate True Positive
     TP = 0;
     positions_is_true_ground_truth_matrix1 = GetSpecificPositionsInMatrix(matrix_ground_truth_datasetTwo, 1)
     for idx, x in enumerate(positions_is_true_ground_truth_matrix1):
        a = int(positions_is_true_ground_truth_matrix1[idx].split(',')[0])
        b = int(positions_is_true_ground_truth_matrix1[idx].split(',')[1])
        if(matrix_ground_truth_datasetTwo[a][b] == cardinality_matrix[a][b]):
         TP = TP + 1;
     
     # Calculate False Positive
     FP = 0
     positions_is_true_cardinality_matrix2 = GetSpecificPositionsInMatrix (cardinality_matrix, 1)
     for idx, x in enumerate(positions_is_true_cardinality_matrix2):
      a = int(positions_is_true_cardinality_matrix2[idx].split(',')[0])
      b = int(positions_is_true_cardinality_matrix2[idx].split(',')[1])
      if(matrix_ground_truth_datasetTwo[a][b] == 0):
       FP = FP + 1;
     
     # Calculate False Negative
     FN = 0
     for idx, x in enumerate(positions_is_true_ground_truth_matrix1):
      a = int(positions_is_true_ground_truth_matrix1[idx].split(',')[0])
      b = int(positions_is_true_ground_truth_matrix1[idx].split(',')[1])
      if(cardinality_matrix[a][b] == 0):
       FN = FN + 1;
     
     precision = TP / (TP + FP)
     recall = TP / (TP + FN)
     f1 = 2 * (precision * recall) / (precision + recall)

     if(precision == 10):
        break;

############################################################# DATASET THREE #################################################################
   
    # Get all column attributes of dataSet three
    datasetThree_attributes = GetAttributeOfColumns(datasetThree)

    # Create Matrix for each macher in dataSet three
    matrix_minimum_combiner_datasetThree = np.zeros((len(datasetThree_attributes),len(mediatedSchema_attributes)))
    matrix_maximum_combiner_datasetThree = np.zeros((len(datasetThree_attributes),len(mediatedSchema_attributes)))
    matrix_average_combiner_datasetThree = np.zeros((len(datasetThree_attributes),len(mediatedSchema_attributes)))

    # Applying combination for dataSet three (Min, Max, Average Combiner)
    for idx, datasetThree_attribute in enumerate(datasetThree_attributes):
     for jdx, mediatedSchema_attribute in enumerate(mediatedSchema_attributes):
        editdistanceResult = round(fuzz.ratio(datasetThree_attribute,mediatedSchema_attribute)/100, 2)
        jarowinklerResult = round(jarowinkler_similarity(datasetThree_attribute, mediatedSchema_attribute), 2)
        word2VecResult = round(1 - spatial.distance.cosine(get_vector(datasetThree_attribute,model), 
                                                           get_vector(mediatedSchema_attribute,model)), 2)

        matrix_minimum_combiner_datasetThree[idx][jdx] = MinimumCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
        matrix_maximum_combiner_datasetThree[idx][jdx] = MaximumCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
        matrix_average_combiner_datasetThree[idx][jdx] = AverageCombiner(editdistanceResult, jarowinklerResult, word2VecResult)
       
        similarities_matrix_datasetTwo = [];
    similarities_matrix_datasetTwo.append(matrix_minimum_combiner_datasetTwo);
    similarities_matrix_datasetTwo.append(matrix_maximum_combiner_datasetTwo);
    similarities_matrix_datasetTwo.append(matrix_average_combiner_datasetTwo);

    # Create Ground Truth Matrix for datasetTwo
    matrix_ground_truth_datasetThree = CreateGroundTruthMatrix(["0,0", "1,1", "2,5", "3,2", "4,3"], len(datasetTwo_attributes), 
                                                  len(mediatedSchema_attributes));

    for matrix in similarities_matrix_datasetTwo:
     DoThreshold(0.5, matrix)
     cardinality_matrix = DoCardinalityDatasetThree(matrix, len(datasetThree_attributes), len(mediatedSchema_attributes))
     
     # Calculate True Positive
     TP = 0;
     positions_is_true_ground_truth_matrix = GetSpecificPositionsInMatrix (matrix_ground_truth_datasetThree, 1)
     for idx, x in enumerate(positions_is_true_ground_truth_matrix):
        a = int(positions_is_true_ground_truth_matrix[idx].split(',')[0])
        b = int(positions_is_true_ground_truth_matrix[idx].split(',')[1])
        if(matrix_ground_truth_datasetThree[a][b] == cardinality_matrix[a][b]):
         TP = TP + 1;
     
     # Calculate False Positive
     FP = 0
     positions_is_true_cardinality_matrix = GetSpecificPositionsInMatrix (cardinality_matrix, 1)
     for idx, x in enumerate(positions_is_true_cardinality_matrix):
      a = int(positions_is_true_cardinality_matrix[idx].split(',')[0])
      b = int(positions_is_true_cardinality_matrix[idx].split(',')[1])
      if(matrix_ground_truth_datasetThree[a][b] == 0):
       FP = FP + 1;
     
     # Calculate False Negative
     FN = 0
     for idx, x in enumerate(positions_is_true_ground_truth_matrix):
      a = int(positions_is_true_ground_truth_matrix[idx].split(',')[0])
      b = int(positions_is_true_ground_truth_matrix[idx].split(',')[1])
      if(cardinality_matrix[a][b] == 0):
       FN = FN + 1;
     
     precision = TP / (TP + FP)
     recall = TP / (TP + FN)
     f1 = 2 * (precision * recall) / (precision + recall)

     print('precision is : ' , precision);
     print('recall is' , recall);
     print('f1 is' , f1);

     if(precision == 10):
        break;


    
def GetAttributeOfColumns(dataset):
    return dataset.columns.tolist();

def MinimumCombiner(editdistanceResult, jarowinklerResult, word2VecResult):
    return min(editdistanceResult, jarowinklerResult, word2VecResult);

def MaximumCombiner(editdistanceResult, jarowinklerResult, word2VecResult):
    return max(editdistanceResult, jarowinklerResult, word2VecResult);

def AverageCombiner(editdistanceResult, jarowinklerResult, word2VecResult):
    numbers = [editdistanceResult, jarowinklerResult, word2VecResult]
    return sum(numbers)/len(numbers);

def preprocess(s):
    return [i.lower() for i in s.split()];

def get_vector(s,model):
    return np.sum(np.array([model[i] for i in preprocess(s)]), axis=0);

def GetSpecificPositionsInMatrix (matrix ,number):
    positions=[]
    for indx, arrays in enumerate(matrix):
        for indx1, array in enumerate(arrays):
            if(array == number):
                positions.append(str(indx) + "," + str(indx1))
                 
    return positions;

def DoCardinalityDatasetOne(matrix, rowCount , columnCount):
    cardinality_matrix = np.zeros((rowCount, columnCount))

    # "0,0", "1,1", "2,2", "3,3", "4,4", "5,5", "6,6", "7,7"
    # We know what row and column
    row = max(matrix[0]);
    column = max([row[0] for row in matrix]);
    cardinality_matrix[0][0] = max(row, column)
    
    row = max(matrix[1]);
    column = max([row[1] for row in matrix]);
    cardinality_matrix[1][1] = max(row, column)
    
    row = max(matrix[2]);
    column = max([row[2] for row in matrix]);
    cardinality_matrix[2][2] = max(row, column)
    
    row = max(matrix[3]);
    column = max([row[3] for row in matrix]);
    cardinality_matrix[3][3] = max(row, column)
    
    row = max(matrix[4]);
    column = max([row[4] for row in matrix]);
    cardinality_matrix[4][4] = max(row, column)
    
    row = max(matrix[5]);
    column = max([row[5] for row in matrix]);
    cardinality_matrix[5][5] = max(row, column)
    
    row = max(matrix[6]);
    column = max([row[6] for row in matrix]);
    cardinality_matrix[6][6] = max(row, column)
    
    row = max(matrix[7]);
    column = max([row[7] for row in matrix]);
    cardinality_matrix[7][7] = max(row, column)

    return cardinality_matrix;

def DoCardinalityDatasetTwo(matrix, rowCount, columnCount):
    cardinality_matrix = np.zeros((rowCount, columnCount))

    # We know what row and column
    row = max(matrix[0]);
    column = max([row[0] for row in matrix]);
    cardinality_matrix[0][0] = max(row, column)
    
    row = max(matrix[1]);
    column = max([row[6] for row in matrix]);
    cardinality_matrix[1][6] = max(row, column)
    
    row = max(matrix[2]);
    column = max([row[1] for row in matrix]);
    cardinality_matrix[2][1] = max(row, column)
    
    row = max(matrix[3]);
    column = max([row[3] for row in matrix]);
    cardinality_matrix[3][3] = max(row, column)
    
    row = max(matrix[4]);
    column = max([row[4] for row in matrix]);
    cardinality_matrix[4][4] = max(row, column)
    
    row = max(matrix[5]);
    column = max([row[7] for row in matrix]);
    cardinality_matrix[5][7] = max(row, column)

    return cardinality_matrix;

def DoCardinalityDatasetThree(matrix, rowCount , columnCount):
    cardinality_matrix = np.zeros((rowCount, columnCount))

    # We know what row and column
    row = max(matrix[0]);
    column = max([row[0] for row in matrix]);
    cardinality_matrix[0][0] = max(row, column)
    
    row = max(matrix[1]);
    column =max([row[1] for row in matrix]);
    cardinality_matrix[1][1] = max(row, column)
    
    row = max(matrix[2]);
    column = max([row[4] for row in matrix]);
    cardinality_matrix[2][4] = max(row, column)
    
    row = max(matrix[3]);
    column = max([row[2] for row in matrix]);
    cardinality_matrix[3][2] = max(row, column)
    
    row = max(matrix[4]);
    column = max([row[3] for row in matrix]);
    cardinality_matrix[4][3] = max(row, column)

    return cardinality_matrix;

def DoThreshold(threshold, matrix):
    # Zeroes all values smaller than threshold in the matrix
    matrix[matrix < threshold] = 0;

def CreateGroundTruthMatrix(positions, rowCount , columnCount):
    matrix = np.zeros((rowCount,columnCount))

    # Set positions to 1
    for idx, x in enumerate(positions):
       matrix[int(positions[idx].split(',')[0])][int(positions[idx].split(',')[1])] = 1

    return matrix;

main()
 