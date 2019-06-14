import numpy as np
import psyneulink as pnl
import csv
from random import randint
import sys

if len(sys.argv) < 2:
    print("Error: Need File Name")

fileName = sys.argv[1]

#computer accuracy UDF hidden
# computeAccuracy(trialInformation)
# Inputs: trialInformation[0, 1, 2, 3]
# trialInformation[0] - Task Dimension : [0, 1] or [1, 0]
# trialInformation[1] - Stimulus Dimension: Congruent {[1, 1] or [-1, -1]} // Incongruent {[-1, 1] or [1, -1]}
# trialInformation[2] - Upper Threshold: Probability of DDM choosing upper bound
# trialInformation[3] - Lower Threshold: Probability of DDM choosing lower bound


def readFile(fileName):

    fileContent = []

    # reading csv file
    with open(fileName, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            fileContent.append(row)

        # cast each string into an int
        for i in range(0, len(fileContent)):
            for j in range(0, len(fileContent[0])):
                fileContent[i][j] = int(fileContent[i][j])

    return fileContent


def getTaskTrain(dataFile):
    taskTrain = []
    trials = len(dataFile)
    for i in range(0, trials):
        task = [0, 0]
        if dataFile[i][0] == 1:
            task = [1, 0]
        elif dataFile[i][0] == 2:
            task = [0, 1]

        taskTrain.append(task)

    return taskTrain

def getStimulusTrain():
    a = [-1, 1]
    b = [1, -1]

    stimulusTrain = []
    for i in range(0, 260):
        x = randint(0, 1)
        if x == 0:
            stimulusTrain.append(a)
        elif x == 1:
            stimulusTrain.append(b)
    return stimulusTrain

#def getStimulusTrain(dataFile):
#    
#    stimulusTrain = []
#    trials = len(dataFile)
#    
#    for i in range(0, trials):
#        stimulus = [0, 0]
#        
#        if dataFile[i][2] == 1:
#            stimulus[0] = 1
#        elif dataFile[i][2] == 2:
#            stimulus[0] = -1
#        
#        if dataFile[i][3] == 1:
#            stimulus[1] = 1
#        elif dataFile[i][3] == 2:
#            stimulus[1] = -1
#    
#        stimulusTrain.append(stimulus)
#        
#    return stimulusTrain


def insertCues(tasks, stimuli):
    trials = len(tasks)
    newTasks = []
    newStimuli = []
    for i in range(0, trials):
        newTasks.append(tasks[i])
        newTasks.append(tasks[i])

        newStimuli.append([0, 0])
        newStimuli.append(stimuli[i])

    return newTasks, newStimuli


def extractValues(outputLog):
    decisionVariable = []
    probabilityUpper = []
    probabilityLower = []
    responseTime = []

    DECISION_VARIABLE = outputLog[1][1][4]
    PROBABILITY_LOWER_THRESHOLD = outputLog[1][1][5]
    PROBABILITY_UPPER_THRESHOLD = outputLog[1][1][6]
    RESPONSE_TIME = outputLog[1][1][7]

    for j in range(1, len(PROBABILITY_LOWER_THRESHOLD)):
        decision = DECISION_VARIABLE[j]
        trialUpper = PROBABILITY_UPPER_THRESHOLD[j]
        trialLower = PROBABILITY_LOWER_THRESHOLD[j]
        reaction = RESPONSE_TIME[j]

        decisionVariable.append(decision[0])
        probabilityUpper.append(trialUpper[0])
        probabilityLower.append(trialLower[0])
        responseTime.append(reaction[0])

    return probabilityUpper, probabilityLower

def computeAccuracy(variable):

    taskInputs = variable[0]
    stimulusInputs = variable[1]
    upperThreshold = variable[2]
    lowerThreshold = variable[3]

    accuracy = []
    for i in range(0, len(taskInputs)):

        colorTrial = (taskInputs[i][0] > 0)
        motionTrial = (taskInputs[i][1] > 0)

        # during color trials

        if colorTrial:
            # if the correct answer is the upper threshold
            if stimulusInputs[i][0] > 0:
                accuracy.append(upperThreshold[i])
                # print('Color Trial: 1')

            # if the correct answer is the lower threshold
            elif stimulusInputs[i][0] < 0:
                accuracy.append(lowerThreshold[i])
                # print('Color Trial: -1')

        if motionTrial:
            # if the correct answer is the upper threshold
            if stimulusInputs[i][1] > 0:
                accuracy.append(upperThreshold[i])
                # print('Motion Trial: 1')

            # if the correct answer is the lower threshold
            elif stimulusInputs[i][1] < 0:
                accuracy.append(lowerThreshold[i])
                # print('Motion Trial: -1')

    return accuracy



# BEGIN: Composition Construction

# Constants as defined in Musslick et al. 2018
integrationConstant = 0.8               # Time Constant
DRIFT = 0.25              # Drift Rate
STARTING_POINT = 0.0    # Starting Point
THRESHOLD = 0.05      # Threshold
NOISE = 0.1           # Noise
T0 = 0.2                # T0
congruentWeight = 0.2


# Task Layer: [Color, Motion] {0, 1} Mutually Exclusive
# Origin Node
taskLayer = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                  size=2,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  output_states=[pnl.RESULT],
                                  name='Task Input [I1, I2]')

# Stimulus Layer: [Color Stimulus, Motion Stimulus]
# Origin Node
stimulusInfo = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                     size=2,
                                     function=pnl.Linear(slope=1, intercept=0),
                                     output_states=[pnl.RESULT],
                                     name="Stimulus Input [S1, S2]")

congruenceWeighting = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                          size = 2,
                                          function=pnl.Linear(slope=congruentWeight, intercept= 0),
                                          name = 'Congruence * Automatic Component')

# Activation Layer: [Color Activation, Motion Activation]
# Recurrent: Self Excitation, Mutual Inhibition
# Controlled: Gain Parameter
activation = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
                                            function=pnl.Logistic(gain=1.0),
                                            matrix=[[1.0, -1.0],
                                                    [-1.0, 1.0]],
                                            integrator_mode=True,
                                            integrator_function=pnl.AdaptiveIntegrator(rate=integrationConstant),
                                            initial_value=np.array([[0.0, 0.0]]),
                                            output_states=[pnl.RESULT],
                                            name='Task Activations [Act 1, Act 2]')

# Hadamard product of Activation and Stimulus Information
nonAutomaticComponent = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                              size=2,
                                              function=pnl.Linear(slope=1, intercept=0),
                                              input_states=pnl.InputState(combine=pnl.PRODUCT),
                                              output_states=[pnl.RESULT],
                                              name='Non-Automatic Component [S1*Activity1, S2*Activity2]')

# Summation of nonAutomatic and Automatic Components
ddmCombination = pnl.TransferMechanism(size=1,
                                       function=pnl.Linear(slope=1, intercept=0),
                                       input_states=pnl.InputState(combine=pnl.SUM),
                                       output_states=[pnl.RESULT],
                                       name="Drift = (S1 + S2) + (S1*Activity1 + S2*Activity2)")


decisionMaker = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=DRIFT,
                                                              starting_point=STARTING_POINT,
                                                              threshold=THRESHOLD,
                                                              noise=NOISE,
                                                              t0=T0),
                        output_states=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME,
                                       pnl.PROBABILITY_UPPER_THRESHOLD,
                                       pnl.PROBABILITY_LOWER_THRESHOLD],
                        name='DDM')

taskLayer.set_log_conditions([pnl.RESULT])
stimulusInfo.set_log_conditions([pnl.RESULT])
activation.set_log_conditions([pnl.RESULT, "mod_gain"])
nonAutomaticComponent.set_log_conditions([pnl.RESULT])
ddmCombination.set_log_conditions([pnl.RESULT])
decisionMaker.set_log_conditions([pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD,
                                  pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])


# Composition Creation

stabilityFlexibility = pnl.Composition()

# Node Creation
stabilityFlexibility.add_node(taskLayer)
stabilityFlexibility.add_node(activation)
stabilityFlexibility.add_node(congruenceWeighting)
stabilityFlexibility.add_node(nonAutomaticComponent)
stabilityFlexibility.add_node(stimulusInfo)
stabilityFlexibility.add_node(ddmCombination)
stabilityFlexibility.add_node(decisionMaker)

# Projection Creation
stabilityFlexibility.add_projection(sender=taskLayer, receiver=activation)
stabilityFlexibility.add_projection(sender=activation, receiver=nonAutomaticComponent)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=nonAutomaticComponent)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=congruenceWeighting)
stabilityFlexibility.add_projection(sender=congruenceWeighting, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=nonAutomaticComponent, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=ddmCombination, receiver=decisionMaker)


# Beginning of Controller

# Grid Search Range
# searchRange = pnl.SampleSpec(start=0.25, stop=4.0, num=16)
# searchRange = pnl.SampleSpec(start=1.0, stop=1.9, num=10)

searchRange = pnl.SampleSpec(start=0.5, stop=5.0, num=10)


# Modulate the GAIN parameter from activation layer
# Initalize cost function as 0
signal = pnl.ControlSignal(projections=[(pnl.GAIN, activation)],
                           function=pnl.Linear,
                           variable=1.0,
                           intensity_cost_function=pnl.Linear(slope=0.0),
                           allocation_samples=searchRange)

# Use the computeAccuracy function to obtain selection values
# Pass in 4 arguments whenever computeRewardRate is called
objectiveMechanism = pnl.ObjectiveMechanism(monitor=[taskLayer, stimulusInfo,
                                                     (pnl.PROBABILITY_UPPER_THRESHOLD, decisionMaker),
                                                     (pnl.PROBABILITY_LOWER_THRESHOLD, decisionMaker)],
                                            function=pnl.AccuracyIntegrator,
                                            name="Controller Objective Mechanism")
objectiveMechanism.set_log_conditions(items=pnl.VALUE)

#  Sets trial history for simulations over specified signal search parameters
metaController = pnl.OptimizationControlMechanism(agent_rep=stabilityFlexibility,
                                                  features=[taskLayer.input_state, stimulusInfo.input_state],
                                                  feature_function=pnl.Buffer(history=4),
                                                  name="Controller",
                                                  objective_mechanism=objectiveMechanism,
                                                  function=pnl.GridSearch(),
                                                  control_signals=[signal])

stabilityFlexibility.add_controller(metaController)
stabilityFlexibility.enable_controller = True
stabilityFlexibility.controller_mode = pnl.BEFORE

for i in range(1, len(stabilityFlexibility.controller.input_states)):
    stabilityFlexibility.controller.input_states[i].function.reinitialize()

# END OF COMPOSITION CONSTRUCTION


# TESTING

dataFile = readFile(fileName)
taskTrain = getTaskTrain(dataFile)
stimulusTrain = getStimulusTrain()

taskTrain, stimulusTrain = insertCues(taskTrain, stimulusTrain)

print("File Name:", fileName)
condition = dataFile[0][14]
print("Condition:", condition)

inputs = {taskLayer: taskTrain, stimulusInfo: stimulusTrain}
activation.parameters.value.retain_old_simulation_data = True


stabilityFlexibility.run(inputs)
# activation.log.print_entries()
# print()
# decisionMaker.log.print_entries()
# print()
# objectiveMechanism.log.print_entries()
decisions = decisionMaker.log.nparray()


### OUTPUTS
runs = len(taskTrain)
upper = []
lower = []
accuracy = []
upper, lower = extractValues(decisions)
variable = [taskTrain, stimulusTrain, upper, lower]
accuracy = computeAccuracy(variable)
print(accuracy)


activations = activation.log.nparray()
gains = []
for i in range(0, runs):
    optimalGain = activations[1][1][5][i+1][0]
    gains.append(optimalGain)
print(gains)

print("Buffer Size = 4")
print("Included Cue Trials")


#outFileName = f"out/{fileName}_{condition}_results.csv"
#np.savetxt(outFileName, np.stack((accuracy, gains)))



