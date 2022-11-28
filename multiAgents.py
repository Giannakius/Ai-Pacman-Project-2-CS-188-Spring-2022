# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #anaparastash map with score
        newPos = successorGameState.getPacmanPosition() # tuple
        newFood = successorGameState.getFood() # True-False for food list
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # time that ghost is scared

        "*** YOUR CODE HERE ***"
        
        food = currentGameState.getFood()       # Get all foods (True-False)
        foodList = food.asList()                # and put them on a list
        currentPos = list(successorGameState.getPacmanPosition()) #Current_Position
        distance = float('-inf')

        if (action == 'Stop'):      # action = north-south-east-west OR stop
            return float('-inf')    # exceeds the minimum value

        for current_state in newGhostStates:    # if in current_state exists a ghost and it's not scared.
            if current_state.getPosition() == tuple(currentPos) and (current_state.scaredTimer == 0):
                return float('-inf')    # exceeds the minimum value

        for x in foodList:
            # We find the best distance to our closest dots. 
            # Oso megalyterh einai h timh poy epistrefei h manhattan toso mikroterh einai h arnhtikh ths timh
            # Oso mikroterh manhattan toso megalyterh timh distance . px (100 -> -100 || 6 -> -6 |||| -100 < -6)
            tempDistance = -1 * (manhattanDistance(currentPos, x))

            if (tempDistance > distance):
                distance = tempDistance

        # Ousiastika h synarthsh ayth exei ws stoxo na faei to kontinotero Food , kai gia na to petyxei
        # ayto prepei na epistrefei oso to dynaton megalh times oso pio konta sto Food brisketai.
        return distance


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    # def MaxValue(self , gameState: GameState , depth , temp_agents):
    #     moves = []
    #     if (gameState.isWin() or gameState.isLose() or depth==0):  # if Terminal State
    #          return self.evaluationFunction(GameState)
        
    #     n = float('-inf')                            # agentsIndex = 0 cause we want the Pacman's Actions
    #     actions = gameState.getLegalActions(temp_agents)       # Change to >=1 for Ghosts.          
    #     for x in actions:   # For all possible Pacman's movent options 
            
    #         n = max(n , self.MinValue(gameState.generateSuccessor(0, x) , depth-1 , 1)) # Returns the maximum value of all the X_node childers.
    #         print ("TESTTTTTTTTTT")
    #         moves.extend(x)                                                   # temp_agents is for ghosts min layer of ghost's id number .
    #     return n

    # def MinValue(self , gameState: GameState , depth , temp_agents):
    #     if (gameState.isWin() or gameState.isLose() or depth==1):  # if Terminal State
    #          return self.evaluationFunction(gameState)

    #     n = float('inf')                            # agentsIndex = 0 cause we want the Pacman's Actions
    #     actions = gameState.getLegalActions(temp_agents)       # Change to >=1 for Ghosts.          
    #     for x in actions:   # For all possible Pacman's movent options 
    #         n = min(n , self.MaxValue(gameState.generateSuccessor(0, x) , depth-1,1)) # Returns the maximum value of all the X_node childers.
    #         print ("AEKKKKKKK")
    #     return n

    def miniMax(self, State, depth, aIndex=0):

        if State.isWin() or State.isLose() or depth == 0:
            return (self.evaluationFunction(State),) #return current score
        Agent = State.getNumAgents()
        print ("aIndex = " , aIndex , "Agent = " , Agent)
        if aIndex != Agent - 1:
            nDepth = depth
        else:
            nDepth = depth - 1
        newAIndex = (aIndex + 1) % Agent
        actionLegal=State.getLegalActions(aIndex)
        temp5 = []
        for a in actionLegal:
            temp5.append((self.miniMax(State.generateSuccessor(aIndex, a), nDepth, newAIndex)[0], a))
        z=max(temp5) if aIndex == 0 else min(temp5)
        return z


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"    
        # depth = self.depth * gameState.getNumAgents() # cause for one move for pacman , we have NumAgents-1 moves for Ghosts
        # agent_turn = 0 # 0 = Pacman turn // 1 ... NumAgents-1 = Ghost turn
        # while (depth>=0) :    
        #     if (gameState.isWin() or gameState.isLose() or depth==0):  # if terminal node 
        #         return Directions.STOP
        #     elif (agent_turn == 0 ):  # Max - Pacman's turn
        #         agent_turn += 1
                
                




        #     elif (agent_turn > 0 ):  # Min - Ghost's turn
        #         if ( agent_turn==gameState.getNumAgents()-1 ):     # one more move for the ghosts
        #             agent_turn = 0 
        #         else:                                               # time for pacman to play!
        #             agent_turn += 1


        return self.miniMax(gameState, self.depth)[1]



        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
