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

    def miniMax(self, GameState, depth, temp_Agent=0):

        if GameState.isWin() or GameState.isLose() or depth == 0:   # If terminal_node (Win-Lose-End_of_the_tree)
            return (self.evaluationFunction(GameState),"Stop")      # Return ......   
        
        All_Agents = GameState.getNumAgents()           # Getting the number of all the Agents(Pacman+Ghosts)
        
        # Agents goes 0 for Pacman and 1,2,3,... n for the Ghosts

        if temp_Agent != All_Agents - 1:        # If temp_Agent is the NOT the last Agent.
            temp_Depth = depth                  # initialize temp_Depth to tree depth.  
        else:                                   # Else if its the Last Ghost turn 
            temp_Depth = depth - 1              # Go to the next depth.
        

        next_temp_Agent = (temp_Agent + 1) % All_Agents # next_temp_Agent goes to the next Agent and if temp_Agent was
                                                        #  the last Agent goes again to the Start. PX(agents=3)-> 0,1,2,0,1,2....
        
        actionLegal = GameState.getLegalActions(temp_Agent)     # A list of legal actions for temp_Agent.
        
        all_possible_scores = []
        for action in actionLegal:  # For all the possible moves , find the score of the pacman's route and his last move       # [0] is beacuse its a tuple('score','move') and we want only the score.
            all_possible_scores.append((self.miniMax(GameState.generateSuccessor(temp_Agent, action), temp_Depth, next_temp_Agent)[0], action))
        
        if temp_Agent == 0 :    # if Agent is Pacman , Find the MAX move , ie the MAX possible SCORE.
            move = max(all_possible_scores)
        else:                   # else if Agent is Ghost , Find the MIN move , ie the MIN possible SCORE.
            move = min(all_possible_scores)
        
        return move


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
        
        result = self.miniMax(gameState, self.depth)
        # [1] is beacause we want to return only the last_move , not the score.
        return result[1]



        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        depthstart = 0 
        agentstart = 0 #Pacman

        # Get the action and score for pacman (max)
        action, score = self.alpha_beta(depthstart, agentstart, gameState, float('-inf'), float('inf'))
        
        return action  


    def alpha_beta(self, curr_depth, agent_index, gameState, alpha, beta):
        
        #   Max player-Pacman == (agent_index=0)
        #   Min player-Ghosts == (agent_index!=0)
        #   If alpha > beta, we can stop generating further successors and prune the search tree.

        # If all agents have played - Restart the proccess
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth = curr_depth + 1
        
        # If Terminal node
        if gameState.isWin() or gameState.isLose() or curr_depth == self.depth :
            # Return the Action and Score
            return None, self.evaluationFunction(gameState)
        
        best_score  = None
        best_action = None
        
        if agent_index == 0:  # Pacman's Turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of pacman
            
                next_game_state = gameState.generateSuccessor(agent_index, action)
                # find the next's player moves now
                x , score = self.alpha_beta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
                
                # if you find a better score
                if best_score is None or score > best_score:
                    best_action = action
                    best_score = score
                    
                
                # MAX cause Pacman's turn 
                alpha = max(score , alpha)
                
                # Prune the tree if alpha is greater than beta
                if alpha > beta:
                    break

        else:  # Ghost's Turn 
            for action in gameState.getLegalActions(agent_index):  # For each legal action of ghost agent
                
                next_game_state = gameState.generateSuccessor(agent_index, action)
                x, score = self.alpha_beta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
                
                if best_score is None or score < best_score:
                    best_action = action
                    best_score = score
                
                # MIN cause Ghost's turn 
                beta = min(beta, score)

                # Prune the tree if beta is less than alpha
                if beta < alpha:
                    break

        # If it's a leaf with no successors and score is None
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        else:
        # Return the best_action and best_score
            return best_action, best_score  



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

        action, score = self.ExpectiMax(0, 0, gameState)  # Get the action and score for pacman (agent_index=0)
        return action  # Return the action to be done as per minimax algorithm



    def ExpectiMax(self, curr_depth, agent_index, gameState):
        #   Max player-Pacman == (agent_index=0)
        #   Min player-Ghosts == (agent_index!=0)
        #   If alpha > beta, we can stop generating further successors and prune the search tree.

        # If all agents have played - Restart the proccess
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth = curr_depth + 1
        
        # If Terminal node
        if gameState.isWin() or gameState.isLose() or curr_depth == self.depth :
            # Return the Action and Score
            return None, self.evaluationFunction(gameState)
        
        best_score  = None
        best_action = None
        
        if agent_index == 0:  # Pacman's Turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of pacman
            
                next_game_state = gameState.generateSuccessor(agent_index, action)
                # find the next's player moves now
                x , score = self.ExpectiMax(curr_depth, agent_index + 1, next_game_state)
                
                # if you find a better score
                if best_score is None or score > best_score:
                    best_action = action
                    best_score = score
                    

        else:  # Ghost's Turn 

            ghostActions = gameState.getLegalActions(agent_index)
            if len(ghostActions) is not 0:
                prob = 1 / len(ghostActions)
            
            for action in ghostActions:  # For each legal action of ghost agent
                
                next_game_state = gameState.generateSuccessor(agent_index, action)
                x, score = self.ExpectiMax(curr_depth, agent_index + 1, next_game_state)
                
                if best_score is None :
                    best_score = 0

                best_score = best_score + prob * score
                best_action = action
                

        # If it's a leaf with no successors and score is None
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        else:
        # Return the best_action and best_score
            return best_action, best_score  


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
    game_score = currentGameState.getScore()

    closest_food = 1
    capsule_count = len(currentGameState.getCapsules())

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    

    # Find distances from pacman to all food
    food_distances = []
    for food_position in food_list:
        food_distances.append( manhattanDistance(pacman_position, food_position) )

    # Set value for closest food if there is still food left
    if food_count > 0:
        closest_food = min(food_distances)

    # Find distances from pacman to ghost(s)
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        # If ghost is too close to pacman, prioritize escaping instead of eating the closest food
        # by resetting the value for closest distance to food
        if ghost_distance < 2:
            closest_food = 99999

    features = [1.0 / closest_food, game_score, food_count, capsule_count]
    weights = [10,200,-100,-10]


    Features_multi_Weights=[]
    
    for i in range(0, len(features)):
        Features_multi_Weights.append(features[i]* weights[i])

    return sum(Features_multi_Weights)
    


# Abbreviation
better = betterEvaluationFunction
