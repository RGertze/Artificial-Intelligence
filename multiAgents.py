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
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        best_score = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == best_score]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
       
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        powerPellets = successorGameState.getCapsules()

        # calculate the evaluation score
        score = successorGameState.getScore()

        # calculate the remaining food pellets
        foodLeft = len(newFood.asList())

        # penalize the score based on the number of remaining food pellets
        if foodLeft == 0:
            score += 1000
        else:
            score -= foodLeft

        # calculate distance to closest food pellet
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(foodDistances) > 0:
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 1
        score += 1 / minFoodDistance

        # calculate distance to closest ghost
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances)
        if minGhostDistance == 0:
            # if Pacman is already caught, return a very low score
            return -100000
        if minGhostDistance <= 1 and sum(newScaredTimes) == 0:
            score -= 1000
        elif minGhostDistance <= 2 and sum(newScaredTimes) == 0:
            score -= 500
        elif minGhostDistance <= 3 and sum(newScaredTimes) == 0:
            score -= 100
        else:
            score += 1 / minGhostDistance
            score += sum(newScaredTimes)

        # check if power pellets are available
        if powerPellets:
            powerDistances = [manhattanDistance(newPos, pellet) for pellet in powerPellets]
            minPowerDistance = min(powerDistances)
            if currentGameState.getNumFood() == 20:
                score += 1 / minPowerDistance
            elif currentGameState.getNumFood() == 116:
                score += 2 / minPowerDistance
            else:
                minPowerDistance = float("inf")

        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        # Base cases for minimax function
        # If we've reached the max depth or if we've won or lost, 
        # then return the evaluation function value.
        def minimax(gameState, depth, agentIndex, cache):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState), None
        
            # Caching implememation whereby, in the case we've already encountered the gameState before, 
            # then we can return its cached result from memory.
            if gameState in cache:
                return cache[gameState]
            
            # If we're looking at the max player/Pacman's moves,
            # we want to find the move that will maximize our evaluation function.
            if agentIndex == 0:
                best_action = None
                best_score = float('-inf')

                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = minimax(successor, depth, agentIndex + 1, cache)

                    if score > best_score:
                        best_score = score
                        best_action = action

                result = best_score, best_action

            # If we're looking at the min player/ghosts' moves, 
            # we want to find the move that will minimize our evaluation function.
            else:
                best_action = None
                best_score = float('inf')

                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)

                    if agentIndex == gameState.getNumAgents() - 1:
                        score, _ = minimax(successor, depth - 1, 0, cache)
                    else:
                        score, _ = minimax(successor, depth, agentIndex + 1, cache)

                    if score < best_score:
                        best_score = score
                        best_action = action

                result = best_score, best_action

            # Cache the result to memory for the gameState to speed up future searches.
            cache[gameState] = result

            return result

        cache = {}
        _, action = minimax(gameState, self.depth, 0, cache)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def max_value(gameState, alpha, beta, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            v = float("-inf")
            bestAction = None
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                score = min_value(successor, alpha, beta, depth, 1)
                if score > v:
                    v = score
                    bestAction = action
                if v > beta:
                    return v, bestAction
                alpha = max(alpha, v)
            return v, bestAction

        def min_value(gameState, alpha, beta, depth, ghostIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float("inf")
            for action in gameState.getLegalActions(ghostIndex):
                successor = gameState.generateSuccessor(ghostIndex, action)
                if ghostIndex == gameState.getNumAgents() - 1:
                    score = max_value(successor, alpha, beta, depth + 1)[0]
                else:
                    score = min_value(successor, alpha, beta, depth, ghostIndex + 1)
                if score < v:
                    v = score
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        alpha = float("-inf")
        beta = float("inf")
        _, action = max_value(gameState, alpha, beta, 0)
        return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def max_value(state, depth):
            # Check if we have reached the maximum depth or terminal state
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            # Find the maximum value and its corresponding action
            v = float("-inf")
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                new_v, _ = exp_value(successor, depth, 1)
                if new_v > v:
                    v = new_v
                    max_action = action
            return v, max_action

        def exp_value(state, depth, agent):
            # Check if we have reached the maximum depth or terminal state
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            # Find the expected value and its corresponding action
            v = 0
            actions = state.getLegalActions(agent)
            num_actions = len(actions)
            if agent == 0:  # Pacman's turn
                for action in actions:
                    successor = state.generateSuccessor(agent, action)
                    new_v, _ = exp_value(successor, depth + 1, agent % state.getNumAgents())
                    v += new_v
                return float(v) / num_actions, None
            else:  # Ghost's turn
                for action in actions:
                    successor = state.generateSuccessor(agent, action)
                    new_v, _ = exp_value(successor, depth, agent % state.getNumAgents())
                    v += new_v
                return float(v) / num_actions, None

        _, action = max_value(gameState, 0)
        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

