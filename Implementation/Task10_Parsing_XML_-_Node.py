import numpy as np
from jarowinkler import *

def main():
 attributes_datasetOne=[];
 attributes_datasetTwo=[];
 attributes_datasetThree=[];
 attributes_mediatedSchema=[];

 attributes_datasetOne.append(['Country']);
 attributes_datasetOne.append(['Country','Name']);
 attributes_datasetOne.append(['Country','latitude']);
 attributes_datasetOne.append(['Country','Longitude']);
 attributes_datasetOne.append(['Country','EstablishDate']);
 attributes_datasetOne.append(['Country','OtheInfo']);
 attributes_datasetOne.append(['Country','County']);
 attributes_datasetOne.append(['Employee']);
 attributes_datasetOne.append(['Employee','HasDormitory']);
 attributes_datasetOne.append(['Employee','Address']);
 attributes_datasetOne.append(['Employee','TotalEnrollment']);
 attributes_datasetOne.append(['Employee','CapacityDormitory']);
 attributes_datasetOne.append(['Employee','City']);
 attributes_datasetOne.append(['Employee','City','ZIPCode']);
 attributes_datasetOne.append(['Employee','City','Address2']);
 attributes_datasetOne.append(['Employee','City','StudentCount']);
 attributes_datasetOne.append(['Employee','City','Telephone']);
 attributes_datasetOne.append(['Employee','City','Website']);

 attributes_datasetTwo.append(['UniversityName']);
 attributes_datasetTwo.append(['UniversityName','Url']);
 attributes_datasetTwo.append(['UniversityName','AverageGPA']);
 attributes_datasetTwo.append(['UniversityName','ContactName']);
 attributes_datasetTwo.append(['UniversityName','EmployeeCount']);
 attributes_datasetTwo.append(['UniversityName','CountStudentWorkInUni']);
 attributes_datasetTwo.append(['UniversityName','NumberOfDisabledStudent']);
 attributes_datasetTwo.append(['UniversityName','AnnualScholarShipGranted']);
 attributes_datasetTwo.append(['UniversityName','NumberOfGraduatedStudents']);
 attributes_datasetTwo.append(['UniversityName','County']);
 attributes_datasetTwo.append(['UniversityName','County','ZIP']);
 attributes_datasetTwo.append(['UniversityName','County','Address']);

 attributes_datasetThree.append(['Country']);
 attributes_datasetThree.append(['Country','City']);
 attributes_datasetThree.append(['Country','ContactName']);
 attributes_datasetThree.append(['Country','Location']);
 attributes_datasetThree.append(['Country','Location','Zip']);
 attributes_datasetThree.append(['Country','Location','Description']);
 attributes_datasetThree.append(['Country','UndergradEnrollment']);
 attributes_datasetThree.append(['Country','UndergradEnrollment','Date']);
 attributes_datasetThree.append(['Country','UndergradEnrollment','Uni']);
 attributes_datasetThree.append(['Country','UndergradEnrollment','Rank']);
 attributes_datasetThree.append(['Country','UndergradEnrollment','TuitionFees']);

 attributes_mediatedSchema.append(['UniversityName']);
 attributes_mediatedSchema.append(['UniversityName','Name']);
 attributes_mediatedSchema.append(['UniversityName','Url']);
 attributes_mediatedSchema.append(['UniversityName','Uni']);
 attributes_mediatedSchema.append(['UniversityName','Location']);
 attributes_mediatedSchema.append(['UniversityName','TotalEmployee']);
 attributes_mediatedSchema.append(['UniversityName','EmployeeCount']);
 attributes_mediatedSchema.append(['UniversityName','Country']);
 attributes_mediatedSchema.append(['UniversityName','Country','City']);
 attributes_mediatedSchema.append(['UniversityName','Country','Address']);
 attributes_mediatedSchema.append(['UniversityName','Country','ZIPCode']);
 attributes_mediatedSchema.append(['UniversityName','Country','County']);
 attributes_mediatedSchema.append(['UniversityName','Country','Website']);

# Create Matrix for each macher in datasetOne 
 matrix = np.zeros((len(attributes_datasetOne),len(attributes_mediatedSchema)))
 
 # Applying combination for Dataset One (Min, Max, Average Combiner)
 for idx, attribute in enumerate(attributes_datasetOne):
     for jdx, mediatedSchema_attribute in enumerate(attributes_mediatedSchema):
        jarowinklerResult = round(jaccard_similarity(attribute, mediatedSchema_attribute),3)
        matrix[idx][jdx]=jarowinklerResult

 print('matrix_datasetOne')
 print(matrix)

 # Create Ground Truth Matrix for datasetOne
 datasetOneTruth=["0,7", "1,1", "6,11", "9,9", "12,8", "13,10", "17,12"];
 datasetTwoTruth=["0,0", "1,2", "4,6", "9,11", "11,9"];
 datasetThreeTruth=["0,7", "1,8","3,4", "8,3"];
 matrix_ground_truth_datasetOne = CreateGroundTruthMatrix(datasetOneTruth, len(attributes_datasetOne), len(attributes_mediatedSchema));

 print('matrix_ground_truth_datasetOne')
 print(matrix_ground_truth_datasetOne)

 DoThreshold(0.5, matrix)
 cardinality_matrix = DoCardinalityDatasetOne(matrix,len(attributes_datasetOne), len(attributes_mediatedSchema), 1)

 print('cardinality_matrix')
 print(cardinality_matrix)

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
 
 precision = weird_division(TP , (TP + FP))
 recall = weird_division(TP , (TP + FN)) 
 f1 = weird_division(2 * (precision * recall) , (precision + recall))
 
 print('TP =',TP);
 print('FP =',FP);
 print('FN =',FN);
 print('precision is',precision);
 print('precision is',precision);
 print('recall is',recall);
 print('f1-measure is',f1);


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def weird_division(n, d):
    return n / d if d else 0

def CreateGroundTruthMatrix(positions, rowCount, columnCount):
    matrix = np.zeros((rowCount,columnCount))

    # Set positions to 1
    for idx, x in enumerate(positions):
       matrix[int(positions[idx].split(',')[0])][int(positions[idx].split(',')[1])] = 1

    return matrix;

def DoThreshold(threshold, matrix):
    matrix[matrix < threshold] = 0;

def DoCardinalityDatasetOne(matrix, rowCount , columnCount, whichDataset):
    cardinality_matrix = np.zeros((rowCount, columnCount))
    if(whichDataset==1):
     # We know what row and column
     row = max(matrix[0]);
     column = max([row[7] for row in matrix]);
     cardinality_matrix[0][7] = max(row, column)
     
     row = max(matrix[1]);
     column = max([row[1] for row in matrix]);
     cardinality_matrix[1][1] = max(row, column)
     
     row = max(matrix[6]);
     column = max([row[11] for row in matrix]);
     cardinality_matrix[6][11] = max(row, column)
        
     row = max(matrix[9]);
     column = max([row[9] for row in matrix]);
     cardinality_matrix[9][9] = max(row, column)
     
     row = max(matrix[12]);
     column = max([row[8] for row in matrix]);
     cardinality_matrix[12][8] = max(row, column)
     
     row = max(matrix[13]);
     column = max([row[10] for row in matrix]);
     cardinality_matrix[13][10] = max(row, column)
     
     row = max(matrix[17]);
     column = max([row[12] for row in matrix]);
     cardinality_matrix[17][12] = max(row, column)
    if(whichDataset==2):
     row = max(matrix[0]);
     column = max([row[0] for row in matrix]);
     cardinality_matrix[0][0] = max(row, column)
     
     row = max(matrix[1]);
     column = max([row[2] for row in matrix]);
     cardinality_matrix[1][2] = max(row, column)
     
     row = max(matrix[4]);
     column = max([row[6] for row in matrix]);
     cardinality_matrix[4][6] = max(row, column)
        
     row = max(matrix[9]);
     column = max([row[11] for row in matrix]);
     cardinality_matrix[9][11] = max(row, column)
     
     row = max(matrix[11]);
     column = max([row[9] for row in matrix]);
     cardinality_matrix[11][9] = max(row, column)
    if(whichDataset==3):
     row = max(matrix[0]);
     column = max([row[7] for row in matrix]);
     cardinality_matrix[0][7] = max(row, column)
     
     row = max(matrix[1]);
     column = max([row[8] for row in matrix]);
     cardinality_matrix[1][8] = max(row, column)
     
     row = max(matrix[3]);
     column = max([row[4] for row in matrix]);
     cardinality_matrix[3][4] = max(row, column)
        
     row = max(matrix[8]);
     column = max([row[3] for row in matrix]);
     cardinality_matrix[8][3] = max(row, column)

    return cardinality_matrix;

def GetSpecificPositionsInMatrix (matrix ,number):
    positions=[]
    for indx, arrays in enumerate(matrix):
        for indx1, array in enumerate(arrays):
            if(array == number):
                positions.append(str(indx) + "," + str(indx1))
                 
    return positions;

main()