import os
import numpy as np
from thefuzz import fuzz
from jarowinkler import *
import xml.etree.ElementTree as ET

def main():
 datasetOne_path=os.path.dirname(__file__)+'\\XmlFiles\\CollegesUniversities.xml';
 mediatedSchema_path=os.path.dirname(__file__)+'\\XmlFiles\\MediatedSchema.xml';

 tree_datasetOne  = ET.parse(datasetOne_path);
 root_datasetOne = tree_datasetOne.getroot();
 tree_mediatedSchema  = ET.parse(mediatedSchema_path);
 root_mediatedSchema = tree_mediatedSchema.getroot();

 attributes_datasetOne=[];
 attributes_mediatedSchema=[];

 for indx, elem in enumerate(root_datasetOne.iter()):
     attributes_datasetOne.append(elem.tag);

 for indx, elem in enumerate(root_mediatedSchema.iter()):
     attributes_mediatedSchema.append(elem.tag);

 # Create Matrix for each macher in datasetOne 
 matrix_average_combiner_datasetOne = np.zeros((len(attributes_datasetOne),len(attributes_mediatedSchema)))
 
 # Applying combination for Dataset One (Min, Max, Average Combiner)
 for idx, datasetOne_attribute in enumerate(attributes_datasetOne):
     for jdx, mediatedSchema_attribute in enumerate(attributes_mediatedSchema):
        editdistanceResult = fuzz.ratio(datasetOne_attribute,mediatedSchema_attribute)/100
        jarowinklerResult = jarowinkler_similarity(datasetOne_attribute, mediatedSchema_attribute)

        matrix_average_combiner_datasetOne[idx][jdx] = AverageCombiner(editdistanceResult, jarowinklerResult)

 # Create Ground Truth Matrix for datasetOne
 matrix_ground_truth_datasetOne = CreateGroundTruthMatrix(["2,0", "3,1", "5,2", "6,3", "9,4", "10,5", "13,6", "17,7","2,8","13,9","17,10","2,11","6,13"], 
                                                             len(attributes_datasetOne), len(attributes_mediatedSchema));

 DoThreshold(0.5, matrix_average_combiner_datasetOne)
 cardinality_matrix = DoCardinalityDatasetOne(matrix_average_combiner_datasetOne, 
                                              len(attributes_datasetOne), len(attributes_mediatedSchema))
 
 # Calculate True Positive
 TP = 0;
 positions_is_true_ground_truth_matrix = GetSpecificPositionsInMatrix(matrix_ground_truth_datasetOne, 1)
 for idx, x in enumerate(positions_is_true_ground_truth_matrix):
    a = int(positions_is_true_ground_truth_matrix[idx].split(',')[0])
    b = int(positions_is_true_ground_truth_matrix[idx].split(',')[1])
    if(matrix_ground_truth_datasetOne[a][b] == cardinality_matrix[a][b]):
     TP = TP + 1;
 
 # Calculate False Positive
 # Positions that are one in Cardinality matrix
 # And are zero in the Ground Truth matrix
 FP = 0
 positions_is_true_cardinality_matrix = GetSpecificPositionsInMatrix(cardinality_matrix, 1)
 for idx, x in enumerate(positions_is_true_cardinality_matrix):
  a = int(positions_is_true_cardinality_matrix[idx].split(',')[0])
  b = int(positions_is_true_cardinality_matrix[idx].split(',')[1])
  if(matrix_ground_truth_datasetOne[a][b] == 0):
   FP = FP + 1;
 
 # Calculate False Negative
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
 
 print(precision);
 print(recall);
 print(f1);

def AverageCombiner(editdistanceResult, jarowinklerResult):
    numbers = [editdistanceResult, jarowinklerResult]
    return sum(numbers)/len(numbers);

def CreateGroundTruthMatrix(positions, rowCount , columnCount):
    matrix = np.zeros((rowCount,columnCount))

    # Set positions to 1
    for idx, x in enumerate(positions):
       matrix[int(positions[idx].split(',')[0])][int(positions[idx].split(',')[1])] = 1

    return matrix;

def DoThreshold(threshold, matrix):
    # Zeroes all values smaller than threshold in the matrix
    matrix[matrix < threshold] = 0;

def DoCardinalityDatasetOne(matrix, rowCount , columnCount):
    cardinality_matrix = np.zeros((rowCount, columnCount))
    
    # We know what row and column
    row = max(matrix[2]);
    column = max([row[0] for row in matrix]);
    cardinality_matrix[2][0] = max(row, column)
    
    row = max(matrix[3]);
    column = max([row[1] for row in matrix]);
    cardinality_matrix[3][1] = max(row, column)
   
    row = max(matrix[5]);
    column = max([row[2] for row in matrix]);
    cardinality_matrix[5][2] = max(row, column)
    
    row = max(matrix[6]);
    column = max([row[3] for row in matrix]);
    cardinality_matrix[6][3] = max(row, column)
    
    row = max(matrix[9]);
    column = max([row[4] for row in matrix]);
    cardinality_matrix[9][4] = max(row, column)
   
    row = max(matrix[10]);
    column = max([row[5] for row in matrix]);
    cardinality_matrix[10][5] = max(row, column)
    
    row = max(matrix[13]);
    column = max([row[6] for row in matrix]);
    cardinality_matrix[13][6] = max(row, column)
    
    row = max(matrix[17]);
    column = max([row[7] for row in matrix]);
    cardinality_matrix[17][7] = max(row, column)
     
    row = max(matrix[2]);
    column = max([row[8] for row in matrix]);
    cardinality_matrix[2][8] = max(row, column)
   
    row = max(matrix[13]);
    column = max([row[9] for row in matrix]);
    cardinality_matrix[13][9] = max(row, column)

    row = max(matrix[17]);
    column = max([row[10] for row in matrix]);
    cardinality_matrix[17][10] = max(row, column)

    row = max(matrix[2]);
    column = max([row[11] for row in matrix]);
    cardinality_matrix[2][11] = max(row, column)

    row = max(matrix[6]);
    column = max([row[13] for row in matrix]);
    cardinality_matrix[6][13] = max(row, column)

    return cardinality_matrix;

def GetSpecificPositionsInMatrix (matrix ,number):
    positions=[]
    for indx, arrays in enumerate(matrix):
        for indx1, array in enumerate(arrays):
            if(array == number):
                positions.append(str(indx) + "," + str(indx1))
                 
    return positions;

main()