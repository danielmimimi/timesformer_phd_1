"""
************************************************************************************************************************

Generic hslu annotation handling.

------------------------------------------------------------------------------------------------------------------------

 Author      Jonas Hofstetter

 Copyright   CC Intelligent Sensors and Networks
             Lucerne University of Applied Sciences
             and Arts T&A, Switzerland.

************************************************************************************************************************
"""

import csv
import os
from pathlib import Path
import sys
from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, TimeoutError
import csvkit

def write_chunk(annotationFilePath, chunk):
    # print(f"Writing chunk of size {len(chunk)} to {annotationFilePath}...")
    with open(annotationFilePath, 'a') as file:
        writer = csvkit.writer(file)
        writer.writerows(chunk)

class AnnotationHandlerHslu:

    """Generic hslu annotation handling."""

    def __init__(self):

        """The constructor."""

        self.annotationItemsList = None
        self.annotationDefaultHeader_2_1 = []
        self.annotationDefaultHeader_2_1.append('# Annotation File\r\n')
        self.annotationDefaultHeader_2_1.append('# Format: imageName, label, labelId, shape, ..ShappeAttributes..\r\n')
        self.annotationDefaultHeader_2_1.append('#')
        self.annotationDefaultHeader_2_1.append('# Shape Attributes:\r\n')
        self.annotationDefaultHeader_2_1.append('# - rect: topLeftX, topLeftY, width, height\r\n')
        self.annotationDefaultHeader_2_1.append('# - point: x, y\r\n')
        self.annotationDefaultHeader_2_1.append('# - ellipse: xCenter, yCenter, width, height, angle\r\n')
        self.annotationDefaultHeader_2_1.append('# - polygon: x0,y0,x1,y1,...,xn,yn\r\n')
        self.annotationDefaultHeader_2_1.append('# - metaData: key, value\r\n')
        self.annotationDefaultHeader_2_1.append('AnnotationFormatVersion:2.1\r\n')
        self.annotationDefaultHeader_2_1.append('\r\n')


    class annotationItem:
        """Annotation base item structure."""

        def __init__(self, obj=None):
            if obj is None:
                self.file_path = None
                self.imageName = None
                self.imageNameAbs = None
                self.label = None
                self.labelId = None
                self.shape = None
            else:
                self.file_path = obj.file_path
                self.imageName = obj.imageName
                self.imageNameAbs = obj.imageNameAbs
                self.label = obj.label
                self.labelId = obj.labelId
                self.shape = obj.shape

    class annotationItemMetadata(annotationItem):
        """Annotation item metadata structure."""

        def __init__(self, obj=None):
            super().__init__(obj)
            self.shape = 'metadata'
            self.metadataKey = None
            self.metadataValue = None

    class annotationItemRect(annotationItem):
        """Annotation item rect structure."""

        def __init__(self, obj=None):
            super().__init__(obj)
            self.shape = 'rect'
            self.left = None
            self.top = None
            self.width = None
            self.height = None

    class annotationItemPolygon(annotationItem):
        """Annotation item polygon structure."""

        def __init__(self, obj=None):
            super().__init__(obj)
            self.shape = 'polygon'
            self.polygon = []

    class annotationItemPoint(annotationItem):
        """Annotation item polygon structure."""

        def __init__(self, obj=None):
            super().__init__(obj)
            self.shape = 'point'
            self.point = []

    def readAnnotations(self, annotationFilePath, valueType=float):

        """Reads annotations from a given file path."""

        self.annotationItemsList = []
        annotationDataList = self.readAnnotationFile(annotationFilePath)

        for annotation in annotationDataList:

            annotationShape = annotation[3].strip()

            item = self.annotationItem()
            item.file_path = annotationFilePath.replace('\\', r'/')

            item.imageName = annotation[0].strip()
            endName = item.imageName.find('+rot0#')
            if endName > 0:
                item.frameNumber = int(item.imageName[endName + 6:])
                item.imageName = item.imageName[0:endName]
            else:
                item.frameNumber = None

            item.label = annotation[1].strip()
            item.labelId = annotation[2].strip()

            if annotationShape.lower() == 'metadata':
                item = self.annotationItemMetadata(item)
                item.metadataKey = annotation[4].strip()
                item.metadataValue = int(annotation[5:][0])

            elif annotationShape == 'rect':
                item = self.annotationItemRect(item)
                item.left = float(annotation[4].strip())
                item.top = float(annotation[5].strip())
                item.width = float(annotation[6].strip())
                item.height = float(annotation[7].strip())

            elif annotationShape.lower() == 'polygon':
                item = self.annotationItemPolygon(item)
                for ii in range(4, len(annotation), 2):
                    pnt = [float(annotation[ii]), float(annotation[ii+1])]
                    item.polygon.append(pnt)

            elif annotationShape.lower() == 'point':
                item = self.annotationItemPoint(item)
                for ii in range(4, len(annotation), 2):
                    pnt = [float(annotation[ii]), float(annotation[ii+1])]
                    if len(pnt) == 2:
                        item.point = pnt

            else:
                self.printError('Not supported annotation shape: {}'.format(annotationShape))

            self.annotationItemsList.append(item)


    def writeAnnotation(self, annotationFilePath, useHeaderFromExistingFile=False, addWhitespace=False):

        """
        Writes annotations from a given file path.
        - useHeaderFromExistingFile: Uses the header of an already existing annotation file.
        """

        if useHeaderFromExistingFile == False:
            annotationHeader = self.annotationDefaultHeader_2_1
        else:
            annotationHeader = self.getAnnotationHeader(annotationFilePath)

        annotationLinesList = []

        for annotationItem in self.annotationItemsList:

            if isinstance(annotationItem, self.annotationItemMetadata):

                annotationLine = [annotationItem.imageName,
                                                  annotationItem.label,
                                                  annotationItem.labelId,
                                                  annotationItem.shape,
                                                  annotationItem.metadataKey,
                                                  annotationItem.metadataValue]

            elif isinstance(annotationItem, self.annotationItemRect):

                annotationLine = [annotationItem.imageName,
                                                  annotationItem.label,
                                                  annotationItem.labelId,
                                                  annotationItem.shape,
                                                  annotationItem.left,
                                                  annotationItem.top,
                                                  annotationItem.width,
                                                  annotationItem.height]
                
            elif isinstance(annotationItem, self.annotationItemPolygon):
                annotationLine = [annotationItem.imageName,
                                                  annotationItem.label,
                                                  annotationItem.labelId,
                                                  annotationItem.shape]
                annotationLine += annotationItem.polygon 
                
            elif isinstance(annotationItem, self.annotationItemPoint):
                annotationLine = [annotationItem.imageName,
                                                  annotationItem.label,
                                                  annotationItem.labelId,
                                                  annotationItem.shape]
                annotationLine += annotationItem.point 
                 #annotationItem.point                                
            else:
                self.printError('Not supported annotation type.')

            annotationLinesList.append(annotationLine)

        file = open(annotationFilePath, "w", newline='')
        file.write(''.join(annotationHeader))
        file.close()

        if addWhitespace == True:

            # Add whitespace before each entry in order to have ", ".
            # It is not supported by csv.writer.

            for line in range(len(annotationLinesList)):
                lineList = annotationLinesList[line]
                for col in range(len(lineList)):
                    if col > 0:
                        annotationLinesList[line][col] = ' ' + annotationLinesList[line][col]
        # print(len(annotationLinesList))


        # Function to write a chunk of data to the CSV

            # print(f"Finished writing chunk of size {len(chunk)} to {annotationFilePath}.")

            # print(f"Writing chunk of size {len(chunk)} to {annotationFilePath}...")
            # with open(annotationFilePath, 'a', newline='') as file:
            #     writer = csv.writer(file, delimiter=',', skipinitialspace=False, quoting=csv.QUOTE_MINIMAL)
            #     writer.writerows(chunk)
            # print(f"Finished writing chunk of size {len(chunk)} to {annotationFilePath}.")
            # # with open(annotationFilePath, 'a', newline='') as file:
            #     writer = csv.writer(file, delimiter=',', skipinitialspace=False, quoting=csv.QUOTE_MINIMAL)
            #     writer.writerows(chunk)

        def delete_file_again(file):
            file_path = Path(file)
            if file_path.exists():
                file_path.unlink(missing_ok=True)

        valid_data = [line for line in annotationLinesList if isinstance(line, list)]
        timeout = 10
        chunk_size = 500
        with ProcessPoolExecutor(max_workers=1) as executor:
            for i in range(0, len(valid_data), chunk_size):
                chunk = valid_data[i:i + chunk_size]
                future = executor.submit(write_chunk, annotationFilePath, chunk)
                try:
                    # print(f"Submitting chunk {i // chunk_size + 1} for writing...")
                    future.result(timeout=timeout)
                    # print(f"Successfully appended {len(chunk)} lines to {annotationFilePath}.")
                except TimeoutError:
                    delete_file_again(annotationFilePath)
                    print(f"Writing to {annotationFilePath} timed out.")
                    break
                except Exception as e:
                    delete_file_again(annotationFilePath)
                    print(f"An unexpected error occurred: {str(e)}")
                    break
                
        

        # signal.signal(signal.SIGALRM, handler)

        # def write_chunk(chunk):
        #     try:
        #         # signal.alarm(timeout)  # Set the timeout
        #         with open(annotationFilePath, 'a', newline='') as file:
        #             writer = csv.writer(file, delimiter=',', skipinitialspace=False, quoting=csv.QUOTE_MINIMAL)
        #             writer.writerows(chunk)
        #         # signal.alarm(0)  # Reset the alarm
        #         print(f"Successfully appended {len(chunk)} lines to {annotationFilePath}.")
        #     except TimeoutException:
        #         print(f"Writing to {annotationFilePath} timed out.")
        #     except IOError as e:
        #         print(f"An I/O error occurred: {e.strerror}")
        #     except Exception as e:
        #         print(f"An unexpected error occurred: {str(e)}")
        #     finally:
        #         pass
        #         # signal.alarm(0)  # Reset the alarm in case of other exceptions

        # # Validate and chunk data
        # valid_data = [line for line in annotationLinesList if isinstance(line, list)]
        # for i in range(0, len(valid_data), chunk_size):
        #     chunk = valid_data[i:i + chunk_size]
        #     write_chunk(chunk)



        # try:
        #     with open(annotationFilePath, 'a', newline='') as file:
        #         writer = csv.writer(file, delimiter=',', skipinitialspace=False, quoting=csv.QUOTE_MINIMAL)
        #         writer.writerows(annotationLinesList)
        
        # except IOError as e:
        #     print(e)
        #     # with open(annotationFilePath, 'a', newline='') as file:
        #     # writer = csv.writer(file, delimiter=',', skipinitialspace=False,quoting=csv.QUOTE_MINIMAL)
        #     # writer.writerows(annotationLinesList)


    def getAnnotationHeader(self, annotationFilePath):

        if os.path.exists(annotationFilePath) == False:
            self.printError('Annotation file does not exist: {}'.format(annotationFilePath))

        with open(annotationFilePath) as fp:
            annotationLinesList = fp.readlines()
        fp.close()

        versionLineNumber = annotationLinesList.index([annotation for annotation in annotationLinesList
                                                      if annotation.find('AnnotationFormatVersion:') == 0][0])

        annotationHeader = annotationLinesList[0:versionLineNumber + 1]
        annotationHeader.append('\n')

        return annotationHeader


    def getAnnotationItem(self, imageSetsDir, imageName, showWarningsIfNotFound=True):

        """Returns a list of annotation items."""

        if len(self.annotationItemsList) == 0:
            self.printError('No annotation items found, use readAnnotations() first.')

        imageNameAbs = os.path.abspath(os.path.join(imageSetsDir, imageName))
        imageNameAbs = imageNameAbs.replace('\\', r'/')

        itemList = [item for item in self.annotationItemsList if item.imageNameAbs == imageNameAbs]

        if (len(itemList) == 0) & (showWarningsIfNotFound == True):
            self.printWarning('No annotation items found: {} '.format(imageNameAbs))

        if len(itemList) == 0:
            return None
        else:
            return itemList
        
        
    def getAnnotationItemsList(self):
    
        """Returns a list of all annotation items."""
    
        return self.annotationItemsList


    def updateValueOfItem(self, annotationItemUpdate):

        """
        The values of an existing annotation item with data from annotationItemUpdate.

        1. Find item using imageName and shape.
        2. Replace label, labelId, and shape values if not None

        """

        matchIndicesList = [i for i in range(len(self.annotationItemsList))
                         if self.annotationItemsList[i].imageName == annotationItemUpdate.imageName]

        if len(matchIndicesList) != 1:
            self.printError('One entry match expected.')

        self.annotationItemsList[matchIndicesList[0]].label = annotationItemUpdate.label
        self.annotationItemsList[matchIndicesList[0]].labelId = annotationItemUpdate.labelId
        self.annotationItemsList[matchIndicesList[0]].shape = annotationItemUpdate.shape

        if  isinstance(annotationItemUpdate, self.annotationItemMetadata):
            self.metadataKey = annotationItemUpdate.metadataKey
            self.metadataValue = annotationItemUpdate.metadataValue

        elif  isinstance(annotationItemUpdate, self.annotationItemRect):
            self.left = annotationItemUpdate.left
            self.top = annotationItemUpdate.top
            self.width = annotationItemUpdate.width
            self.height = annotationItemUpdate.height

        else:
            self.printError('Not supported annotation type.')


    def readAnnotationsToList(self, imageSetsDir, imageSetDefinitions):

        """Reads all annotations and returns a list."""

        annotationItemsList = []
        annotationHandler = AnnotationHandlerHslu()

        for imageSetItem in imageSetDefinitions:
            annoationPath = os.path.join(str(imageSetItem[0]), imageSetItem[2])

            print('Read annotation: {}'.format(annoationPath))

            annotationFilePath = os.path.join(imageSetsDir, annoationPath)
            annotationDataList = annotationHandler.readAnnotations(annotationFilePath, imageSetItem[0])
            annotationItemsList.extend(annotationDataList)

        # Remove path from recording_id as the names are unique.

        for i in range(len(annotationItemsList)):
            recording_id = os.path.basename(annotationItemsList[i].recording_id)
            annotationItemsList[i].recording_id = recording_id

        return annotationItemsList


    def readAnnotationFile(self, annotationFilePath):

        """Reads all annotations from given file path."""

        if os.path.exists(annotationFilePath) == False:
            self.printError('Annotation file not found: {}'.format(annotationFilePath))

        annotationDataList = self.readCsvToList(annotationFilePath, ',', isUtf8=False)

        return annotationDataList


    def readCsvToList(self, filePath, delimiter, isUtf8=True):

        """Reads a csv list of a given path and returns a list."""

        csvList = []

        if isUtf8 == True:
            csvfile = open(filePath, 'r', encoding='utf-8')
        else:
            csvfile = open(filePath, 'r')

        csvReader = csv.reader(csvfile, delimiter=delimiter)

        for line in csvReader:

            if not line:
                continue
            if line[0][0] == '#':
                continue
            if line[0].find('AnnotationFormatVersion') >= 0:
                continue

            csvList.append(line)

        return csvList


    def printError(self, errorText):

        """Prints the errorText and stops the script."""

        print(f'\033[31;1mERROR: ' + errorText + '\033[31;1m')
        sys.exit(-1)


    def printWarning(self, warningText):

        """Prints the warningText."""

        print(f'\033[33;1mWARNING: ' + warningText + '\033[33;1m')
        print(f'\033[0;1m\033[0;1m')
    
    def get_count_of_elements(self,imageName:str):
        count = 0
        for image in self.annotationItemsList:
            if(image.imageName == imageName):
                count = count + 1
        return count


    def handle_annotation_item_list_creation(self):
        """Creates if none, throws if not list otherwise"""
        if self.annotationItemsList is None:
            self.annotationItemsList = []
        elif not type(self.annotationItemsList) == list:
            raise AssertionError('wrong annotation item list type: {}'.format(type(self.annotationItemsList)))  

    def addRect(self, imageName: str, boundingBoxList: List[Tuple[str, str, str, str]], label: str):
        self.handle_annotation_item_list_creation()

        for boundingBox in boundingBoxList:
            ai = self.annotationItemRect()
            ai.imageName = imageName+":rot0"
            ai.label = '{}'.format(label)
            ai.labelId = str(self.get_count_of_elements(imageName))
            ai.shape = 'rect'
            ai.left, ai.top, ai.width, ai.height = boundingBox
            self.annotationItemsList.append(ai)

    def addMetadata(self,imageName:str,key:str,value:str):
        self.handle_annotation_item_list_creation()

        ai = self.annotationItemMetadata()
        ai.imageName = imageName+":rot0"
        ai.metadataKey = key
        ai.metadataValue = value

        self.annotationItemsList.append(ai) 

    def addPolygon(self,imageName:str,polygon:List[int],label:str,label_id:int=-1):
        self.handle_annotation_item_list_creation()

        ai = self.annotationItemPolygon()
        ai.imageName = imageName+":rot0"
        ai.label = '{}'.format(label)
        if label_id == -1:
            ai.labelId = str(self.get_count_of_elements(imageName))
        else:
            ai.labelId = label_id
        ai.shape = 'polygon'
        ai.polygon = polygon
        self.annotationItemsList.append(ai)

    def addPoint(self,imageName:str,point:List[int],label:str,label_id:int=-1):
        self.handle_annotation_item_list_creation()
        if len(point) == 2:
            ai = self.annotationItemPoint()
            ai.imageName = imageName+":rot0"
            ai.label = '{}'.format(label)
            if label_id == -1:
                ai.labelId = str(self.get_count_of_elements(imageName))
            else:
                ai.labelId = label_id
            ai.shape = 'point'
            ai.point = point
            self.annotationItemsList.append(ai)
        else:
            raise Exception("Use Polygon for more than 1 point")