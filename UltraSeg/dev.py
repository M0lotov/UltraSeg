import os
import logging
import numpy as np
import time
import vtk, qt, slicer

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import torch
from Resources.unext import UNext
from Resources.UtilConnections import UtilConnections
import torch.backends.cudnn as cudnn
import cv2

#
# dev
#

class dev(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "dev"
    self.parent.categories = ["Ultrasound"]
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Yanlin Chen"]
    self.parent.helpText = """Ultrasound segmentation using UNet in real time."""
    self.parent.acknowledgementText = """
SUSTech CS 330 - Multimedia Information Processing Course Project
"""

#
# devWidget
#

class devWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None

    self.inputModifiedObserverTag = None

    self.outputImageNode = None  # For observation
    self.lastFpsUpdateTime = 0  # To prevent too fast (hard to see) label update on GUI
    self.fpsLabelCooldown_s = 0.5

    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)


    # Load widget from .ui file (created by Qt Designer)

    uiWidget = slicer.util.loadUI(self.resourcePath('UI/dev.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create a new parameterNode (it stores user's node and parameter values choices in the scene)
    self.logic = devLogic()
    self.setParameterNode(self.logic.getParameterNode())

    if not self._parameterNode.GetNodeReferenceID('Segmentation'):
      segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Segmentation')
      segmentationNode.CreateDefaultDisplayNodes()
      segmentationNode.GetSegmentation().AddEmptySegment('nerve')
      segmentationNode.GetSegmentation().AddEmptySegment('vessel')
      self._parameterNode.SetNodeReferenceID("Segmentation", segmentationNode.GetID())
    

    if not self._parameterNode.GetNodeReferenceID('Prompt'):
      promptNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Prompt')
      promptNode.CreateDefaultDisplayNodes()
      promptNode.GetSegmentation().AddEmptySegment('nerve')
      promptNode.GetSegmentation().AddEmptySegment('vessel')
      promptNode.GetDisplayNode().SetAllSegmentsVisibility2DFill(False)
      self._parameterNode.SetNodeReferenceID("Prompt", promptNode.GetID())
    

    # Connections

    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneImportEnd)

    self.ui.modelPathLineEdit.connect("currentPathChanged(QString)", self.onModelSelected)
    lastModelPath = self.logic.getLastModelPath()
    if lastModelPath is not None:
      self.ui.modelPathLineEdit.setCurrentPath(lastModelPath)

    self.ui.checkBox_unext.connect('stateChanged(int)', self.onUnextChecked)
    self.ui.checkBox_sam.connect('stateChanged(int)', self.onSAMChecked)
    
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputImageSelected)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputImageSelected)
    # self.ui.browserSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onBrowserSelected)
    # self.ui.outputTransformComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputTransformSelected)

    self.ui.applyButton.connect('toggled(bool)', self.onApplyButton)
    # self.ui.applyButton.enabled = False
    # self.ui.embeddingButton.connect('clicked(bool)', self.onEmbeddingButton)

    self.ui.SegmentEditorWidget.setMRMLSegmentEditorNode(slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentEditorNode', 'MySegmentEditor'))
    self.ui.SegmentEditorWidget.setSegmentationNode(slicer.util.getNode('Prompt'))
    # self.ui.SegmentEditorWidget.setActiveEffectByName('Scissors')
    effect = self.ui.SegmentEditorWidget.effectByName('Scissors')
    effect.setOperation(2)
    effect.setShape(2)

    # Initial GUI update
    self.updateGUIFromParameterNode()

    self.logic.connections = UtilConnections()
    self.logic.connections.setup()

    self.logic.setModel('sam')


  def onSceneImportEnd(self, caller, event):
    parameterNode = self.logic.getParameterNode()
    parameterNode.Modified()

  def enter(self):
    pass
    # slicer.util.setApplicationLogoVisible(False)
    # slicer.util.setDataProbeVisible(False)

  def exit(self):
    pass
    # slicer.util.setApplicationLogoVisible(True)
    # slicer.util.setDataProbeVisible(True)

  def onUnextChecked(self, checked):
    if checked==2:
      self.ui.checkBox_sam.blockSignals(True)
      self.ui.checkBox_sam.setCheckState(0)
      self.ui.checkBox_sam.blockSignals(False)
      self.ui.modelPathLineEdit.enabled = True
      # self.ui.embeddingButton.enabled = False
      self.ui.SegmentEditorWidget.enabled = False
      self.logic.setModel('unext')

  def onSAMChecked(self, checked):
    if checked==2:
      self.ui.checkBox_unext.blockSignals(True)
      self.ui.checkBox_unext.setCheckState(0)
      self.ui.checkBox_unext.blockSignals(False)
      self.ui.modelPathLineEdit.enabled = False
      self.ui.SegmentEditorWidget.enabled = True
      self.logic.setModel('sam')

  def onInputImageSelected(self, selectedNode):
    self.logic.setInputImage(selectedNode)
    self.ui.SegmentEditorWidget.setSourceVolumeNode(selectedNode)

  def onOutputImageSelected(self, selectedNode):
    self.logic.setOutputImage(selectedNode)

  def onBrowserSelected(self, selectedNode):
    self.logic.setBrowser(selectedNode)

  # def onOutputTransformSelected(self, selectedNode):
  #   self.logic.setOutputTransform(selectedNode)


  def onModelSelected(self, modelFullname):
    self.logic.setModelPath(modelFullname)

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()
    if self.logic.connections:
        self.logic.connections.clear()

  def setParameterNode(self, inputParameterNode):
    """
    Adds observers to the selected parameter node. Observation is needed because when the
    parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)
    # wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
    # self.ui.parameterNodeSelector.setCurrentNode(inputParameterNode)
    # self.ui.parameterNodeSelector.blockSignals(wasBlocked)
    if inputParameterNode == self._parameterNode:
      return
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    if inputParameterNode is not None:
      self.addObserver(inputParameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    # Disable all sections if no parameter node is selected

    self.ui.basicCollapsibleButton.enabled = self._parameterNode is not None

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    self._updatingGUIFromParameterNode = True

    # Update each widget from parameter node

    # wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
    # self.ui.parameterNodeSelector.setCurrentNode(self._parameterNode)
    # self.ui.parameterNodeSelector.blockSignals(wasBlocked)

    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.INPUT_IMAGE))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_IMAGE))
    # self.ui.browserSelector.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.SEQUENCE_BROWSER))
    # self.ui.outputTransformComboBox.setCurrentNode(self._parameterNode.GetNodeReference(self.logic.OUTPUT_TRANSFORM))

    self.ui.applyButton.checked = self.logic.getPredictionActive()

    # Update buttons states and tooltips

    if self._parameterNode.GetNodeReference(self.logic.INPUT_IMAGE) and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output volume"
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"

    self._updatingGUIFromParameterNode = False

  def onOutputModified(self, caller, event):
    """
    Updates status label text.
    :param caller:
    :param event:
    :return: None
    """
    if self.ui.applyButton.checked == True:
      if (time.time() - self.lastFpsUpdateTime) >= self.fpsLabelCooldown_s:
        self.ui.feedbackLabel.text = "Prediction running at {:.1f} FPS".format(self.logic.getFps())
        self.lastFpsUpdateTime = time.time()
    else:
      self.ui.feedbackLabel.text = "Prediction stopped"

  def onEmbeddingButton(self):
    try:
      self.logic.computeEmbedding()
      self.ui.applyButton.enabled = True
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute embedding: "+str(e))
      import traceback
      traceback.print_exc()
      
  def onApplyButton(self, toggled):
    """
    Run processing when user clicks "Apply" button.
    """
    self.inputImageNode = self.ui.inputSelector.currentNode()
    if self.inputImageNode is None:
      self.ui.feedbackLabel.text = "Input image not selected!"
      logging.info("Apply button clicked without input selection")
      return
    else:
      logging.info("Input image: {}".format(self.inputImageNode.GetName()))

    self.outputImageNode = self.ui.outputSelector.currentNode()
    if self.outputImageNode is None:
      self.ui.feedbackLabel.text = "Output image not selected!"
      logging.info("Apply button clicked without output selection")
      return
    else:
      logging.info("Output image: {}".format(self.outputImageNode.GetName()))


    if self.ui.modelPathLineEdit.enabled:
      if self.logic.unext_model is None:
        self.ui.feedbackLabel.text = "UNext model not selected!"
        logging.info("Apply button clicked without UNext model selected")
        return
      else:
        logging.info("Using UNext")

    try:
      if toggled:
        self.ui.inputSelector.enabled = False
        self.ui.outputSelector.enabled = False
        # self.ui.browserSelector.enabled = False
        # self.ui.outputTransformComboBox.enabled = False
        self.addObserver(self.outputImageNode, slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onOutputModified)
        self.ui.feedbackLabel.text = "Prediction starting"
      else:
        self.removeObserver(self.outputImageNode, slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onOutputModified)
        self.ui.inputSelector.enabled = True
        self.ui.outputSelector.enabled = True
        # self.ui.browserSelector.enabled = True
        # self.ui.outputTransformComboBox.enabled = True
        self.ui.feedbackLabel.text = "Prediction stopped"

      self.logic.setRealTimePrediction(toggled)
    except Exception as e:
      slicer.util.errorDisplay("Failed to start live segmentation: "+str(e))
      import traceback
      traceback.print_exc()



#
# devLogic
#

class devLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  INPUT_IMAGE = "InputImage"
  OUTPUT_IMAGE = "OutputImage"
  # OUTPUT_TRANSFORM = "OutputTransform"
  OUTPUT_FPS = "OutputFps"
  PREDICTION_ACTIVE = "PredictionActive"
  WAIT_FOR_NODE = "WaitForNode"  # Experimental idea: drop predictions until e.g. volume reconstruction output is updated
  AI_MODEL_FULLPATH = "AiModelFullpath"
  LAST_AI_MODEL_PATH_SETTING = "dev/LastAiModelPath"
  SEQUENCE_BROWSER = 'SequenceBrowser'

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    VTKObservationMixin.__init__(self)

    self.slicer_to_model_scaling = None
    self.model_to_slicer_scaling = None

    self.unext_model = None
    self.apply_logarithmic_transformation = True
    self.logarithmic_transformation_decimals = 4

    self.inputModifiedObserverTag = None

    self.outputLastTime_s = 0
    self.fpsBuffer = np.zeros(5)
    self.waitForNodeLastMTime = 0

    self.predictionPaused = False

    self.connections = None

    self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


  def setDefaultParameters(self, parameterNode):
    if not parameterNode.GetParameter(self.OUTPUT_FPS):
      parameterNode.SetParameter(self.OUTPUT_FPS, "0.0")
    if not parameterNode.GetParameter(self.PREDICTION_ACTIVE):
      parameterNode.SetParameter(self.PREDICTION_ACTIVE, "False")

  def pausePrediction(self):
    """
    Blocks prediction computation until resumePrediction() is called, or prediction is started again.
    """
    self.predictionPaused = True

  def resumePrediction(self):
    """
    Unblocks prediction computation, so prediction will be computed automatically when input is modified.
    """
    self.predictionPaused = False

  def setWaitForNode(self, selectedNode):
    parameterNode = self.getParameterNode()
    if selectedNode is None:
      parameterNode.SetNodeReferenceID(self.WAIT_FOR_NODE, None)
    else:
      parameterNode.SetNodeReferenceID(self.WAIT_FOR_NODE, selectedNode.GetID())
    self.waitForNodeLastMTime = 0  # Use output in first round

  def setModel(self, model):
    parameterNode = self.getParameterNode()
    parameterNode.SetParameter('model', model)

  def setInputImage(self, inputImageNode):
    """
    Sets input image node
    :param inputImageNode: vtkMRMLScalarVolumeNode
    :return: None
    """
    parameterNode = self.getParameterNode()
    if inputImageNode is None:
      parameterNode.SetNodeReferenceID(self.INPUT_IMAGE, None)
      return

    parameterNode.SetNodeReferenceID(self.INPUT_IMAGE, inputImageNode.GetID())

  def setOutputImage(self, outputImageNode):
    """
    Sets output image node
    :param outputImageNode: vtkMRMLScalarVolumeNode
    :return: None
    """
    paramterNode = self.getParameterNode()
    if outputImageNode is None:
      paramterNode.SetNodeReferenceID(self.OUTPUT_IMAGE, None)
      return
    paramterNode.SetNodeReferenceID(self.OUTPUT_IMAGE, outputImageNode.GetID())

  def setBrowser(self, browserNode):
    paramterNode = self.getParameterNode()
    if browserNode is None:
      paramterNode.SetNodeReferenceID(self.SEQUENCE_BROWSER, None)
      return
    paramterNode.SetNodeReferenceID(self.SEQUENCE_BROWSER, browserNode.GetID())

  def setModelPath(self, modelFullpath):
    """
    Sets the AI model file full path
    :param modelFullpath: str
    :return: None
    """
    parameterNode = self.getParameterNode()
    if modelFullpath == "" or modelFullpath is None:
      parameterNode.SetParameter(self.AI_MODEL_FULLPATH, "")
      return
    parameterNode.SetParameter(self.AI_MODEL_FULLPATH, modelFullpath)

    try:
      cudnn.benchmark = True
      model = UNext(num_classes=2).cuda()
      checkpoint = torch.load(modelFullpath)
      model.load_state_dict(checkpoint)
      model.eval()
      self.unext_model = model
      logging.info(self.unext_model)
      logging.info("Model loaded from file: {}".format(modelFullpath))
      settings = qt.QSettings()
      settings.setValue(self.LAST_AI_MODEL_PATH_SETTING, modelFullpath)
    except Exception as e:
      logging.error("Could not load model from file: {}".format(modelFullpath))
      logging.error("Exception: {}".format(str(e)))

  def getBox(self, mask):
      non_zero = np.nonzero(mask)
      x1, x2, y1, y2 = np.min(non_zero[1]), np.max(non_zero[1]), np.min(non_zero[0]), np.max(non_zero[0])
      return np.array([x1,y1,x2,y2], dtype=int)


  def setRealTimePrediction(self, toggled):
    inputImageNode = self.getParameterNode().GetNodeReference(self.INPUT_IMAGE)
    outputImageNode = self.getParameterNode().GetNodeReference(self.OUTPUT_IMAGE)
    if inputImageNode is None:
      logging.error("Cannot start live prediction, input image not specified")
      return

    if toggled == True:
      if self.getParameterNode().GetParameter('model') == 'sam':
        promptNode = self.getParameterNode().GetNodeReference('Prompt')
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'label')
        str_arr = vtk.vtkStringArray()
        str_arr.InsertNextValue('vessel')
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(promptNode, str_arr, labelmapVolumeNode, inputImageNode)
        mask = slicer.util.arrayFromVolume(labelmapVolumeNode)
        mask = mask.squeeze()[::-1, :]
        # print(mask.shape)
        self.vessel_box = self.getBox(mask)

        str_arr = vtk.vtkStringArray()
        str_arr.InsertNextValue('nerve')
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(promptNode, str_arr, labelmapVolumeNode, inputImageNode)
        mask = slicer.util.arrayFromVolume(labelmapVolumeNode)
        mask = mask.squeeze()[::-1, :]
        self.nerve_box = self.getBox(mask)

        input_array = np.squeeze(slicer.util.array(inputImageNode.GetID())).astype(np.uint8)
        input_array = cv2.cvtColor(input_array, cv2.COLOR_GRAY2RGB)
        self.connections.sendCmd({'msg': 'initialize', 'img': input_array[::-1, :], 'prompt': np.array([self.nerve_box, self.vessel_box])})
        self.connections.recv()

        promptNode.SetDisplayVisibility(False)


      self.predictionPaused = False
      self.inputModifiedObserverTag = inputImageNode.AddObserver(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent,
                                                                 self.onInputNodeModified)
      
      self.onInputNodeModified(None, None)  # Compute prediction for current image instead of waiting for an update

    else:
      logging.info("Stopping live segmentation")
      if self.inputModifiedObserverTag is not None:
        inputImageNode.RemoveObserver(self.inputModifiedObserverTag)
        self.inputModifiedObserverTag = None

  def computeEmbedding(self):
    browserNode = self.getParameterNode().GetNodeReference(self.SEQUENCE_BROWSER)
    seqenceNode = browserNode.GetSequenceNode(self.getParameterNode().GetNodeReference(self.INPUT_IMAGE))
    img_size = slicer.util.arrayFromVolume(self.getParameterNode().GetNodeReference(self.INPUT_IMAGE)).shape[1:]
    len_seq = seqenceNode.GetNumberOfDataNodes()
    seq_data = np.zeros((len_seq, *img_size), dtype=np.uint8)
    for i in range(len_seq):
      seq_data[i,...] = np.squeeze(slicer.util.arrayFromVolume(seqenceNode.GetNthDataNode(i)))[::-1, :]
    self.connections.sendCmd({'msg': 'compute embedding', 'data': seq_data})
    msg = self.connections.recv()
    logging.info(msg)


  def updateOutputFps(self):
    """Call this function exactly once per every output prediction generated."""
    currentTime_s = time.time()
    lapsedTime_s = currentTime_s - self.outputLastTime_s
    if self.outputLastTime_s != 0 and lapsedTime_s != 0:
      currentFps = 1.0 / lapsedTime_s
      self.fpsBuffer = np.roll(self.fpsBuffer, 1)
      self.fpsBuffer[0] = currentFps
    self.outputLastTime_s = currentTime_s

  def getPredictionActive(self):
    """
    Returns the state of prediction.
    :return: bool, True for live prediction, False for paused state
    """
    parameterNode = self.getParameterNode()
    if parameterNode.GetParameter(self.PREDICTION_ACTIVE).lower() == "true":
      return True
    else:
      return False

  def getFps(self):
    """
    Returns the average FPS across the last three outputs.
    :return: float, FPS frame/seconds
    """
    return np.mean(self.fpsBuffer)

  def getLastModelPath(self):
    return slicer.util.settingsValue(self.LAST_AI_MODEL_PATH_SETTING, None)

  def onInputNodeModified(self, caller, event):
    """
    Callback function for input image modified event.
    :returns: None
    """
    if self.predictionPaused == True:
      return

    self.updatePrecition()

  def updatePrecition(self):
    parameterNode = self.getParameterNode()
    inputImageNode = parameterNode.GetNodeReference(self.INPUT_IMAGE)
    outputImageNode = parameterNode.GetNodeReference(self.OUTPUT_IMAGE)
    if parameterNode.GetParameter('model') == 'sam':
      nerve_mask, vessel_mask = self.samPredict()
    else:
      nerve_mask, vessel_mask = self.unextPredict()

    nerve_mask = find_max_region(nerve_mask)
    vessel_mask = find_max_region(vessel_mask)
    # segmentationNode = parameterNode.GetNodeReference('Segmentation')
    # slicer.util.updateSegmentBinaryLabelmapFromArray(nerve_mask[::-1, :], segmentationNode, segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('nerve'), inputImageNode)
    # slicer.util.updateSegmentBinaryLabelmapFromArray(vessel_mask[::-1, :], segmentationNode, segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('vessel'), inputImageNode)

    slicer.util.updateVolumeFromArray(outputImageNode, nerve_mask[::-1, :] + vessel_mask[::-1, :])
    
    self.updateOutputFps()  # Update FPS data

  def samPredict(self):
    parameterNode = self.getParameterNode()
    inputImageNode = parameterNode.GetNodeReference(self.INPUT_IMAGE)

    # browserNode = parameterNode.GetNodeReference(self.SEQUENCE_BROWSER)
    # seq_num = browserNode.GetSelectedItemNumber()
    # nerve_box = self.nerve_kf.predict(self.nerve_box)
    # vessel_box = self.vessel_kf.predict(self.vessel_box)
    # print(nerve_box, vessel_box)
    # self.connections.sendCmd({'msg': 'predict', 'seq_num': seq_num, 'boxes': np.array([nerve_box[:4], vessel_box[:4]])})
    # nerve_mask, vessel_mask = self.connections.recv()
    # self.nerve_box = self.getBox(nerve_mask)
    # self.vessel_box = self.getBox(vessel_mask)

    input_array = np.squeeze(slicer.util.array(inputImageNode.GetID())).astype(np.uint8)
    input_array = cv2.cvtColor(input_array, cv2.COLOR_GRAY2RGB)
    self.connections.sendCmd({'msg': 'track', 'img': input_array[::-1, :]})
    nerve_mask, vessel_mask = self.connections.recv()

    return nerve_mask, vessel_mask

  def unextPredict(self):
    parameterNode = self.getParameterNode()
    inputImageNode = parameterNode.GetNodeReference(self.INPUT_IMAGE)
    input_array = np.squeeze(slicer.util.array(inputImageNode.GetID())).astype(np.uint8)
    input_array = self.clahe.apply(input_array)
    input_array = cv2.cvtColor(input_array, cv2.COLOR_GRAY2RGB)
    input_size = input_array.shape[:2][::-1]
    input_array = cv2.resize(input_array, (512, 512), interpolation=cv2.INTER_LINEAR)
    input_tensor = torch.Tensor(input_array.transpose(2,0,1) / 255).cuda()
    with torch.no_grad():
      output = self.unext_model(input_tensor[None,...])
      output = torch.sigmoid(output.squeeze()).cpu().numpy()
      output[output>=0.5]=1
      output[output<0.5]=0
      output = output.astype(np.uint8)
    output = cv2.resize(output.transpose(1,2,0), input_size, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
    nerve_mask, vessel_mask = output[0], output[1]
    return nerve_mask[::-1,:], vessel_mask[::-1,:]
  

def find_max_region(mask_sel):
    mask_sel = np.ascontiguousarray(mask_sel)
    contours, _ = cv2.findContours(mask_sel,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    if len(area) >= 1:
      max_idx = np.argmax(area)
    
      for k in range(len(contours)):
          if k != max_idx:
              cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

#
# devTest
#

class devTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_dev1()

  def test_dev1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    inputVolume = SampleData.downloadFromURL(
      nodeNames='MRHead',
      fileNames='MR-Head.nrrd',
      uris='https://github.com/Slicer/SlicerTestingData/releases/download/MD5/39b01631b7b38232a220007230624c8e',
      checksums='MD5:39b01631b7b38232a220007230624c8e')[0]
    self.delayDisplay('Finished with download and loading')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 279)

    self.delayDisplay('Test passed')


