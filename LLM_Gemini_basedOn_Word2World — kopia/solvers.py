from collections import deque
import threading
import time
import types


class WorldState:
    def __init__(self, walkable_tiles, interactive_tiles, game_map, objective_coordinates):
        self.solid = []
        self.deadlocks = []
        self.targets = []  # Will be set based on objective coordinates
        self.crates = []  # Will be based on interactive tiles
        self.player = None
        self.walkable_tiles = walkable_tiles
        self.interactive_tiles = interactive_tiles
        self.game_map = game_map
        self.objective_coordinates = objective_coordinates

    def stringInitialize(self, lines, objective_coordinates):
        self.solid = []
        self.targets = []
        self.crates = []
        self.player = None

        self.height = len(lines)
        self.width = max(len(line) for line in lines)

        for coords in objective_coordinates:
            self.targets.append({"x": coords[1], "y": coords[0]})

        for y, line in enumerate(lines):
            self.solid.append([])
            for x, c in enumerate(line):
                
                if (c not in self.walkable_tiles) and (c not in self.interactive_tiles) and (c != "@"):
                    self.solid[y].append(True)  # Mark as solid
                else:
                    self.solid[y].append(False)
                    if c in self.interactive_tiles:
                        self.crates.append({"x": x, "y": y})
                    if c == "@":
                        self.player = {"x": x, "y": y}

        self.initializeDeadlocks()

        return self

    def clone(self):
        clone=WorldState(self.walkable_tiles, self.interactive_tiles, self.game_map, self.objective_coordinates)
        clone.width = self.width
        clone.height = self.height
        # since the solid is not changing then copy by value
        clone.solid = self.solid
        clone.deadlocks = self.deadlocks
        clone.player={"x":self.player["x"], "y":self.player["y"]}

        for t in self.targets:
            clone.targets.append({"x":t["x"], "y":t["y"]})

        for c in self.crates:
            clone.crates.append({"x":c["x"], "y":c["y"]})

        return clone
    
    def initializeDeadlocks(self):
        sign = lambda x: int(x/max(1,abs(x)))
        
        self.deadlocks = []
        for y in range(self.height):
            self.deadlocks.append([])
            for x in range(self.width):
                self.deadlocks[y].append(False)
                
        corners = []
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1 or self.solid[y][x]:
                    continue
                if (self.solid[y-1][x] and self.solid[y][x-1]) or (self.solid[y-1][x] and self.solid[y][x+1]) or (self.solid[y+1][x] and self.solid[y][x-1]) or (self.solid[y+1][x] and self.solid[y][x+1]):
                    if not self.checkTargetLocation(x, y):
                        corners.append({"x":x, "y":y})
                        self.deadlocks[y][x] = True
        
        for c1 in corners:
            for c2 in corners:
                dx,dy = sign(c1["x"] - c2["x"]), sign(c1["y"] - c2["y"])
                if (dx == 0 and dy == 0) or (dx != 0 and dy != 0):
                    continue
                walls = []
                x,y=c2["x"],c2["y"]
                if dx != 0:
                    x += dx
                    while x != c1["x"]:
                        if self.checkTargetLocation(x,y) or self.solid[y][x] or (not self.solid[y-1][x] and not self.solid[y+1][x]):
                            walls = []
                            break
                        walls.append({"x":x, "y":y})
                        x += dx
                if dy != 0:
                    y += dy
                    while y != c1["y"]:
                        if self.checkTargetLocation(x,y) or self.solid[y][x] or (not self.solid[y][x-1] and not self.solid[y][x+1]):
                            walls = []
                            break
                        walls.append({"x":x, "y":y})
                        y += dy
                for w in walls:
                    self.deadlocks[w["y"]][w["x"]] = True
    
    def checkDeadlock(self):
        for c in self.crates:
            if self.deadlocks[c["y"]][c["x"]]:
                return True
        return False
    
    def checkOutside(self, x, y):
        return x < 0 or y < 0 or x > len(self.solid[0]) - 1 or y > len(self.solid) - 1

    def checkTargetLocation(self, x, y):
        for t in self.targets:
            if t["x"] == x and t["y"] == y:
                return t
        return None

    def checkCrateLocation(self, x, y):
        for c in self.crates:
            if c["x"] == x and c["y"] == y:
                return c
        return None

    def checkMovableLocation(self, x, y):
        print(f"x: {x}, y: {y}")
        aa = not self.checkOutside(x, y)
        bb = not self.solid[y][x]
        cc = self.checkCrateLocation(x,y)
        print(f"aa = {aa}, bb = {bb}, cc = {cc}")
        return not self.checkOutside(x, y) and not self.solid[y][x] and self.checkCrateLocation(x,y) is None

    def checkWin(self):
        if len(self.targets) != len(self.crates) or len(self.targets) == 0 or len(self.crates) == 0:
            return False

        for t in self.targets:
            if self.checkCrateLocation(t["x"], t["y"]) is None:
                return False

        return True

    def getHeuristic(self):
        if not self.targets:
            return 0  # Return a default value if there are no targets

        distance = 0
        for c in self.crates:
            bestDist = float('inf')
            bestMatch = None
            for i, t in enumerate(self.targets):
                dist = abs(c["x"] - t["x"]) + abs(c["y"] - t["y"])
                if dist < bestDist:
                    bestMatch = i
                    bestDist = dist

            # Only update distance if a best match was found
            if bestMatch is not None:
                distance += abs(self.targets[bestMatch]["x"] - c["x"]) + abs(self.targets[bestMatch]["y"] - c["y"])
                # Optionally, remove the target from consideration if it's matched (depends on the problem's context)
                # del self.targets[bestMatch]

        return distance

    def update(self, dirX, dirY):
        if abs(dirX) > 0 and abs(dirY) > 0:
            return
        if self.checkWin():
            return
        if dirX > 0:
            dirX=1
        if dirX < 0:
            dirX=-1
        if dirY > 0:
            dirY=1
        if dirY < 0:
            dirY=-1
        newX=self.player["x"]+dirX
        newY=self.player["y"]+dirY
        if self.checkMovableLocation(newX, newY):
            self.player["x"]=newX
            self.player["y"]=newY
        else:
            crate=self.checkCrateLocation(newX,newY)
            if crate is not None:
                crateX=crate["x"]+dirX
                crateY=crate["y"]+dirY
                if self.checkMovableLocation(crateX,crateY):
                    self.player["x"]=newX
                    self.player["y"]=newY
                    crate["x"]=crateX
                    crate["y"]=crateY
                    return True
        return False

    def getKey(self):
        key=str(self.player["x"]) + "," + str(self.player["y"]) + "," + str(len(self.crates)) + "," + str(len(self.targets))
        for c in self.crates:
            key += "," + str(c["x"]) + "," + str(c["y"]);
        for t in self.targets:
            key += "," + str(t["x"]) + "," + str(t["y"]);
        return key

    def __str__(self):
        result = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.solid[y][x]:
                    result += "#"
                else:
                    crate=self.checkCrateLocation(x,y) is not None
                    target=self.checkTargetLocation(x,y) is not None
                    player=self.player["x"]==x and self.player["y"]==y
                    if crate:
                        if target:
                            result += "*"
                        else:
                            result += "$"
                    elif player:
                        if target:
                            result += "+"
                        else:
                            result += "@"
                    else:
                        if target:
                            result += "."
                        else:
                            result += " "
            result += "\n"
        return result[:-1]

class Node:
    balance = 0.5
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0
        if self.parent != None:
            self.depth = parent.depth + 1

    def getChildren(self):
        children = []
        directions = [{"x":-1, "y":0}, {"x":1, "y":0}, {"x":0, "y":-1}, {"x":0, "y":1}]
        for d in directions:
            childState = self.state.clone()
            crateMove = childState.update(d["x"], d["y"])
            if childState.player["x"] == self.state.player["x"] and childState.player["y"] == self.state.player["y"]:
                continue
            if crateMove and childState.checkDeadlock():
                continue
            children.append(Node(childState, self, d))
        return children

    def getKey(self):
        return self.state.getKey()

    def getCost(self):
        return self.depth

    def getHeuristic(self):
        return self.state.getHeuristic()

    def checkWin(self):
        return self.state.checkWin()

    def getActions(self):
        actions = []
        current = self
        while(current.parent != None):
            actions.insert(0,current.action)
            current = current.parent
        return actions

    def __str__(self):
        return str(self.depth) + "," + str(self.state.getHeuristic()) + "\n" + str(self.state)
    
    def __lt__(self, other):
        return self.getHeuristic()+Node.balance*self.getCost() < other.getHeuristic()+Node.balance*other.getCost()

class EnhancedAStarWorldAgent():
    def __init__(self, walkable_tiles, objective_tiles, state, important_tiles):
        self.walkable_tiles = walkable_tiles
        self.objective_tiles = set(objective_tiles)
        self.important_tiles = important_tiles 
        self.state = state

    def is_walkable(self, y, x):
        return self.state.game_map[x][y] in self.walkable_tiles

    def find_closest_objective(self, current_position):
        closest_objective = None
        min_distance = float('inf')
        for objective in self.objective_tiles:
            distance = abs(current_position[0] - objective[0]) + abs(current_position[1] - objective[1])
            if distance < min_distance:
                min_distance = distance
                closest_objective = objective
        return closest_objective

    def find_most_common_walkable_tile(self):
        tile_counts = {}
        for row in self.state.game_map:
            for tile in row:
                if tile in self.walkable_tiles:
                    tile_counts[tile] = tile_counts.get(tile, 0) + 1
        return max(tile_counts, key=tile_counts.get)

    def modify_path_to_objective(self, start, end, common_tile):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy  # error value e_xy

        while True:
            #print(x0,y0)
            #print(x1,y1)
            # Only change the tile if it's not walkable, not an objective, and not important
            if (self.state.game_map[y0][x0] not in self.walkable_tiles and 
                self.state.game_map[y0][x0] not in self.important_tiles and 
                (x0, y0) not in self.objective_tiles and
                self.state.game_map[y0][x0] != "@"):
                self.state.game_map[y0][x0] = common_tile

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 >= dy:  # e_xy+e_x > 0
                err += dy
                x0 += sx
            if e2 <= dx:  # e_xy+e_y < 0
                err += dx
                y0 += sy
            
    def getSolution(self, state, balance=1, maxIterations=-1, maxTime=-1):
        start_time = time.perf_counter()
        iterations = 0
        bestNode = None
        Node.balance = balance
        queue = PriorityQueue()
        queue.put(Node(state.clone(), None, None))
        visited = set()
        best_path = []
        max_placed = 0
        check = False
        while (iterations < maxIterations or maxIterations <= 0) and (time.perf_counter() - start_time < maxTime or maxTime < 0) and not queue.empty():
            iterations += 1
            print(f"AStar iterations number: {iterations}")
            current = queue.get()
            current_pos = (current.state.player['x'], current.state.player['y'])

            if current_pos in self.objective_tiles:
                self.objective_tiles.remove(current_pos)

            if not self.objective_tiles:
                return current.getActions(), current, iterations

            if current.getKey() not in visited:
                visited.add(current.getKey())
                children = current.getChildren()
                any_walkable = False

                for c in children:
                    if self.is_walkable(c.state.player['x'], c.state.player['y']):
                        any_walkable = True
                        queue.put(c)
                
                #if not any_walkable:# and children:
                #    
                #    closest_objective = self.find_closest_objective(current_pos)
                #    most_common_tile = self.find_most_common_walkable_tile()
                #    if closest_objective:
                #        self.modify_path_to_objective(current_pos, closest_objective, most_common_tile)
                #        any_walkable = True

                if len(current.getActions()) > max_placed:
                    max_placed = len(current.getActions())
                    best_path = current.getActions()

        print(f"{best_path}, {bestNode}, {iterations}, {current.state.game_map}, {queue.empty()}")
        return best_path, bestNode, iterations, current.state.game_map, queue.empty()

class Full(Exception):
    'Exception raised by Queue.put(block=0)/put_nowait().'
    pass

class Empty(Exception):
        'Exception raised by Queue.get(block=0)/get_nowait().'
        pass

class Queue:
    '''Create a queue object with a given maximum size.

    If maxsize is <= 0, the queue size is infinite.
    '''

    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self._init(maxsize)

        # mutex must be held whenever the queue is mutating.  All methods
        # that acquire mutex must release it before returning.  mutex
        # is shared between the three conditions, so acquiring and
        # releasing the conditions also acquires and releases mutex.
        self.mutex = threading.Lock()

        # Notify not_empty whenever an item is added to the queue; a
        # thread waiting to get is notified then.
        self.not_empty = threading.Condition(self.mutex)

        # Notify not_full whenever an item is removed from the queue;
        # a thread waiting to put is notified then.
        self.not_full = threading.Condition(self.mutex)

        # Notify all_tasks_done whenever the number of unfinished tasks
        # drops to zero; thread waiting to join() is notified to resume
        self.all_tasks_done = threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        '''Indicate that a formerly enqueued task is complete.

        Used by Queue consumer threads.  For each get() used to fetch a task,
        a subsequent call to task_done() tells the queue that the processing
        on the task is complete.

        If a join() is currently blocking, it will resume when all items
        have been processed (meaning that a task_done() call was received
        for every item that had been put() into the queue).

        Raises a ValueError if called more times than there were items
        placed in the queue.
        '''
        with self.all_tasks_done:
            unfinished = self.unfinished_tasks - 1
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times')
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished

    def join(self):
        '''Blocks until all items in the Queue have been gotten and processed.

        The count of unfinished tasks goes up whenever an item is added to the
        queue. The count goes down whenever a consumer thread calls task_done()
        to indicate the item was retrieved and all work on it is complete.

        When the count of unfinished tasks drops to zero, join() unblocks.
        '''
        with self.all_tasks_done:
            while self.unfinished_tasks:
                self.all_tasks_done.wait()

    def qsize(self):
        '''Return the approximate size of the queue (not reliable!).'''
        with self.mutex:
            return self._qsize()

    def empty(self):
        '''Return True if the queue is empty, False otherwise (not reliable!).

        This method is likely to be removed at some point.  Use qsize() == 0
        as a direct substitute, but be aware that either approach risks a race
        condition where a queue can grow before the result of empty() or
        qsize() can be used.

        To create code that needs to wait for all queued tasks to be
        completed, the preferred technique is to use the join() method.
        '''
        with self.mutex:
            return not self._qsize()

    def full(self):
        '''Return True if the queue is full, False otherwise (not reliable!).

        This method is likely to be removed at some point.  Use qsize() >= n
        as a direct substitute, but be aware that either approach risks a race
        condition where a queue can shrink before the result of full() or
        qsize() can be used.
        '''
        with self.mutex:
            return 0 < self.maxsize <= self._qsize()

    def put(self, item, block=True, timeout=None):
        '''Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until a free slot is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot
        is immediately available, else raise the Full exception ('timeout'
        is ignored in that case).
        '''
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def get(self, block=True, timeout=None):
        '''Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until an item is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately
        available, else raise the Empty exception ('timeout' is ignored
        in that case).
        '''
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            self.not_full.notify()
            return item

    def put_nowait(self, item):
        '''Put an item into the queue without blocking.

        Only enqueue the item if a free slot is immediately available.
        Otherwise raise the Full exception.
        '''
        return self.put(item, block=False)

    def get_nowait(self):
        '''Remove and return an item from the queue without blocking.

        Only get an item if one is immediately available. Otherwise
        raise the Empty exception.
        '''
        return self.get(block=False)

    # Override these methods to implement other queue organizations
    # (e.g. stack or priority queue).
    # These will only be called with appropriate locks held

    # Initialize the queue representation
    def _init(self, maxsize):
        self.queue = deque()

    def _qsize(self):
        return len(self.queue)

    # Put a new item in the queue
    def _put(self, item):
        self.queue.append(item)

    # Get an item from the queue
    def _get(self):
        return self.queue.popleft()

    __class_getitem__ = classmethod(types.GenericAlias)


class PriorityQueue(Queue):
    '''Variant of Queue that retrieves open entries in priority order (lowest first).

    Entries are typically tuples of the form:  (priority number, data).
    '''

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        heappush(self.queue, item)

    def _get(self):
        return heappop(self.queue)
    
def heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)

def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt

def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)