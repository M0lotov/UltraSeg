import os
import unittest
import logging
import numpy as np
import pickle
import time
from PIL import Image
import vtk, qt, ctk, slicer

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import torch
import torchvision.transforms
from Resources.unet import UNet
import torch.nn.functional as F
#
# UltraSeg
#

class UltraSeg(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "UltraSeg"
    self.parent.categories = ["Ultrasound"]
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Yanlin Chen"]
    self.parent.helpText = """Ultrasound segmentation using UNet in real time."""
    self.parent.acknowledgementText = """
SUSTech CS 330 - Multimedia Information Processing Course Project
"""

#
# UltraSegWidget
#

class UltraSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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

    uiWidget = slicer.util.loadUI(self.resourcePath('UI/UltraSeg.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create a new parameterNode (it stores user's node and parameter values choices in the scene)
    self.logic = UltraSegLogic()

    self.setParameterNode(self.logic.getParameterNode())

    # Connections

    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneImportEnd)

    self.ui.modelPathLineEdit.connect("currentPathChanged(QString)", self.onModelSelected)
    lastModelPath = self.logic.getLastModelPath()
    if lastModelPath is not None:
      self.ui.modelPathLineEdit.setCurrentPath(lastModelPath)
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputImageSelected)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputImageSelected)
    # self.ui.outputTransformComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputTransformSelected)

    self.ui.applyButton.connect('toggled(bool)', self.onApplyButton)

    # Initial GUI update
    self.updateGUIFromParameterNode()

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

  def onInputImageSelected(self, selectedNode):
    self.logic.setInputImage(selectedNode)

  def onOutputImageSelected(self, selectedNode):
    self.logic.setOutputImage(selectedNode)

  # def onOutputTransformSelected(self, selectedNode):
  #   self.logic.setOutputTransform(selectedNode)


  def onModelSelected(self, modelFullname):
    self.logic.setModelPath(modelFullname)

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

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

    if self.logic.unet_model is None:
      self.ui.feedbackLabel.text = "UNet model not selected!"
      logging.info("Apply button clicked without UNet model selected")
      return
    else:
      logging.info("Using UNet")

    try:
      if toggled:
        self.ui.inputSelector.enabled = False
        self.ui.outputSelector.enabled = False
        # self.ui.outputTransformComboBox.enabled = False
        self.addObserver(self.outputImageNode, slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onOutputModified)
        self.ui.feedbackLabel.text = "Prediction starting"
      else:
        self.removeObserver(self.outputImageNode, slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent, self.onOutputModified)
        self.ui.inputSelector.enabled = True
        self.ui.outputSelector.enabled = True
        # self.ui.outputTransformComboBox.enabled = True
        self.ui.feedbackLabel.text = "Prediction stopped"

      self.logic.setRealTimePrediction(toggled)
    except Exception as e:
      slicer.util.errorDisplay("Failed to start live segmentation: "+str(e))
      import traceback
      traceback.print_exc()



#
# UltraSegLogic
#

class UltraSegLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
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
  LAST_AI_MODEL_PATH_SETTING = "UltraSeg/LastAiModelPath"

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    VTKObservationMixin.__init__(self)

    self.slicer_to_model_scaling = None
    self.model_to_slicer_scaling = None

    self.unet_model = None
    self.apply_logarithmic_transformation = True
    self.logarithmic_transformation_decimals = 4

    self.inputModifiedObserverTag = None

    self.outputLastTime_s = 0
    self.fpsBuffer = np.zeros(5)
    self.waitForNodeLastMTime = 0

    self.predictionPaused = False

    self.vesselLabelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'Vessel')
    self.nerveLabelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'Nerve')
    self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'UltraSoundSegmentation')
    self.segmentationNode.GetSegmentation().AddEmptySegment('nerve')
    self.segmentationNode.GetSegmentation().AddEmptySegment('vessel')

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

  # def setOutputTransform(self, selectedNode):
  #   parameterNode = self.getParameterNode()
  #   if selectedNode is None:
  #     parameterNode.SetNodeReferenceID(self.OUTPUT_TRANSFORM, None)
  #     return
  #   parameterNode.SetNodeReferenceID(self.OUTPUT_TRANSFORM, selectedNode.GetID())

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
      model = UNet(n_channels=1, n_classes=3).cuda()
      checkpoint = torch.load(modelFullpath)
      model.load_state_dict(checkpoint)
      model.eval()
      self.unet_model = model
      logging.info(self.unet_model)
      logging.info("Model loaded from file: {}".format(modelFullpath))
      settings = qt.QSettings()
      settings.setValue(self.LAST_AI_MODEL_PATH_SETTING, modelFullpath)
    except Exception as e:
      logging.error("Could not load model from file: {}".format(modelFullpath))
      logging.error("Exception: {}".format(str(e)))

  def setRealTimePrediction(self, toggled):
    inputImageNode = self.getParameterNode().GetNodeReference(self.INPUT_IMAGE)
    outputImageNode = self.getParameterNode().GetNodeReference(self.OUTPUT_IMAGE)
    if inputImageNode is None:
      logging.error("Cannot start live prediction, input image not specified")
      return

    if self.unet_model is None:
      logging.error("Cannot start live prediction, AI model is missing")
      return

    if toggled == True:

      input_array = slicer.util.array(inputImageNode.GetID())  # (Z, F, M)
      input_array = input_array[0, :, :]  # (F, M)

      # Start observing input image

      self.predictionPaused = False
      self.inputModifiedObserverTag = inputImageNode.AddObserver(slicer.vtkMRMLScalarVolumeNode.ImageDataModifiedEvent,
                                                                 self.onInputNodeModified)
      self.onInputNodeModified(None, None)  # Compute prediction for current image instead of waiting for an update
      # slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(outputImageNode.GetID())
      # slicer.app.layoutManager().sliceWidget("Yellow").mrmlSliceNode().SetOrientation('Axial')
      # slicer.app.layoutManager().sliceWidget("Yellow").sliceController().fitSliceToBackground()

    else:
      logging.info("Stopping live segmentation")
      if self.inputModifiedObserverTag is not None:
        inputImageNode.RemoveObserver(self.inputModifiedObserverTag)
        self.inputModifiedObserverTag = None

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
    input_array = slicer.util.array(inputImageNode.GetID())
    # logging.debug(input_array.dtype)
    # input_array axis directions (Z, F, M):
    # 1: out of plane = Z
    # 2: sound direction = F
    # 3: transducer mark direction = M

    # input_image = Image.fromarray(input_array[0, :, :])  # image.width is M, image.height is F
    # resized_input_array = np.array(  # np.array will be (F, M) again, but resize happens in (M, F) axis order
    #   input_image.resize(
    #     (
    #       512, #int(input_image.width * self.slicer_to_model_scaling[1]),  # M direction (width on US machine)
    #       512, #int(input_image.height * self.slicer_to_model_scaling[0]),  # F direction (height on US machine)
    #     ),
    #     resample=Image.BILINEAR
    #   )
    # )
    # resized_input_array = np.flip(resized_input_array, axis=0)  # Flip to trained sound direction
    # resized_input_array = resized_input_array / resized_input_array.max()  # Scaling intensity to 0-1
    # resized_input_array = np.expand_dims(resized_input_array, axis=0)  # Add Batch dimension
    # resized_input_array = np.expand_dims(resized_input_array, axis=3)
    input_tensor = torchvision.transforms.Compose([torchvision.transforms.Resize((512,512)), torchvision.transforms.ToTensor()])(Image.fromarray(input_array[0,:,:])).unsqueeze(0).cuda()
    output_tensor = self.unet_model(input_tensor)

    # output_array = y[0, :, :, 1]  # Remove batch dimension (F, M)
    # output_array = np.flip(output_array, axis=0)  # Flip back to match input sound direction

    # apply_logarithmic_transformation = True
    # logarithmic_transformation_decimals = 4
    # if apply_logarithmic_transformation:
    #   e = logarithmic_transformation_decimals
    #   output_array = np.log10(np.clip(output_array, 10 ** (-e), 1.0) * (10 ** e)) / e
    # output_image = Image.fromarray(output_array)  # F -> height, M -> width
    # upscaled_output_array = np.array(
    #   output_image.resize(
    #     (
    #       int(output_image.width * self.model_to_slicer_scaling[1]),
    #       int(output_image.height * self.model_to_slicer_scaling[0]),
    #     ),
    #     resample=Image.BILINEAR,
    #   )
    # )
    # upscaled_output_array = upscaled_output_array * 255
    # upscaled_output_array = np.clip(upscaled_output_array, 0, 255)
    # logging.debug(torch.unique(torch.argmax(output_tensor, dim=1)))
    output_tensor[:,1:,...][output_tensor[:,1:,...]<0.99] = 0
    output_cls = torch.argmax(output_tensor, dim=1).cpu().numpy().astype(np.uint8)
    output_cls = np.asarray(Image.fromarray(output_cls[0,:,:]).resize((input_array.shape[2], input_array.shape[1]), resample=Image.BILINEAR))[np.newaxis,...]
    # logging.debug(output_cls.shape)
    output_array = input_array.copy()
    output_array[output_cls==0] = 0
    nerve_mask = (output_cls==1).astype(np.uint8)
    vessel_mask = (output_cls==2).astype(np.uint8)
    # logging.debug(np.unique(input_array))
    # logging.debug('out')
    # logging.debug(np.unique(output_array))
    # if 127 in output_array:
    #   logging.debug(np.unique(output_array))
    #   Image.fromarray(output_array[0,:,:]).save('D:\\downloads\\test.png')
    # output = Image.fromarray(output_array[0,:,:]).convert('L')
    # logging.debug(output.getcolors())
    # output_array[output_array==1] = 127
    # output_array[output_array==2] = 255
    # logging.debug(np.unique(output_array))


    # VolumeIJKToRAS = vtk.vtkMatrix4x4()
    # inputImageNode.GetIJKToRASMatrix(VolumeIJKToRAS)
    # self.vesselLabelNode.SetIJKToRASMatrix(VolumeIJKToRAS)
    # self.nerveLabelNode.SetIJKToRASMatrix(VolumeIJKToRAS)  

    # slicer.util.updateVolumeFromArray(self.nerveLabelNode, nerve_mask)
    # slicer.util.updateVolumeFromArray(self.vesselLabelNode, vessel_mask)
    slicer.util.updateSegmentBinaryLabelmapFromArray(nerve_mask, self.segmentationNode, self.segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('nerve'), inputImageNode)
    slicer.util.updateSegmentBinaryLabelmapFromArray(vessel_mask, self.segmentationNode, self.segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('vessel'), inputImageNode)
    # self.nerveSegNode.CreateClosedSurfaceRepresentation()
    # outputImageNode = parameterNode.GetNodeReference(self.OUTPUT_IMAGE)
    # slicer.util.updateVolumeFromArray(outputImageNode, upscaled_output_array.astype(np.uint8)[np.newaxis, ...])
    # slicer.util.updateVolumeFromArray(outputImageNode, output_array)
    # Update output transform, just to be compatible with running separate process

    # inputImageNode = parameterNode.GetNodeReference(self.INPUT_IMAGE)
    # imageTransformNode = inputImageNode.GetParentTransformNode()
    # outputTransformNode = parameterNode.GetNodeReference(self.OUTPUT_TRANSFORM)
    # if imageTransformNode is not None and outputTransformNode is not None:
    #   inputTransformMatrix = vtk.vtkMatrix4x4()
    #   imageTransformNode.GetMatrixTransformToWorld(inputTransformMatrix)
    #   outputTransformNode.SetMatrixTransformToParent(inputTransformMatrix)

    self.updateOutputFps()  # Update FPS data


#
# UltraSegTest
#

class UltraSegTest(ScriptedLoadableModuleTest):
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
    self.test_UltraSeg1()

  def test_UltraSeg1(self):
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


