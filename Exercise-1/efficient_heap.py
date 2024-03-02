class HeapNode:
    def __init__(self, value, key, identifier):
        self.value = value
        self.key = key
        self.identifier = identifier


class EfficientHeap:

    def __init__(self, validate=False):
        """
        :param validate : If set to True after every operation the state of the heap
        is checked for testing purposes. Makes the queue's operations O(N^2).
        """
        self._items = []
        self._validate = validate
        self._item_positions = {}
        self._length = 0
        self._free_identifier = 0

    def __len__(self):
        return self._length

    def __check_valid(self):
        """
        Raises an exception if the item ids do not match or self.__length
        doesn't match self.__items and self.__item_positions lengths.
        """
        for i in range(1, self._length):
            if (self._items[i].key <
                    self._items[self.parent_pos(i)].key):
                raise Exception(f'item at {i} invalid')

        for i in self._item_positions:
            if self._items[self._item_positions[i]].identifier != i:
                raise Exception(f'item at {self._item_positions[i]} invalid')

        if self._length != len(self._item_positions):
            raise Exception(f'invalid')

    def pop(self):
        """
        Returns the first element in the heap and removes it in O(log(N)) time.
        :return: value of the element, key of the element
        """
        r = self._items[0]
        self.remove(r.identifier)
        return r.value, r.key

    def remove(self, identifier):
        """
        Removes an element with a given identifier.
        """
        pos = self._item_positions[identifier]
        # Swap object to be removed with the furthest object to maintain continuity.
        self.swap(pos, self._length - 1)
        # Decrease length to remove the object to be removed.
        self._length -= 1
        # While the former furthest object has children and the smaller child is smaller than the
        # former furthest object swap with child.
        while True:
            scp = self.smaller_child_pos(pos)
            if (self.smaller_child_pos(pos) < self._length and
                    (self._items[scp].key <
                     self._items[pos].key)):
                self.swap(pos, scp)
                pos = scp
                continue
            else:
                break
        self._item_positions.pop(identifier)
        if self._validate:
            self.__check_valid()

    def swap(self, pos1, pos2):
        """
        Swap the objects at given positions and update self._item_positions to reflect this.
        """
        self._items[pos1], self._items[pos2] = self._items[pos2], self._items[pos1]
        self._item_positions[self._items[pos2].identifier] = pos2
        self._item_positions[self._items[pos1].identifier] = pos1

    @staticmethod
    def parent_pos(pos):
        return (pos - 1) // 2

    def smaller_child_pos(self, pos):
        """
        :param pos: Position to get the smaller child of.
        :return: The smaller child or the left child if any of the children don't exist.
        """
        if self.left_child_pos(pos) >= self._length or self.right_child_pos(pos) >= self._length:
            return self.left_child_pos(pos)
        if self._items[self.left_child_pos(pos)].key < self._items[self.right_child_pos(pos)].key:
            return self.left_child_pos(pos)
        else:
            return self.right_child_pos(pos)

    @staticmethod
    def left_child_pos(pos):
        return pos * 2 + 1

    @staticmethod
    def right_child_pos(pos):
        return pos * 2 + 2

    def add(self, value, key):
        """
        Adds a new value in O(log(N)) time.
        :param value: The value the element is to have.
        :param key: The key the element is to be stored under.
        :return: The element's unique identifier to be used in other operations.
        """
        identifier = self._free_identifier
        self._free_identifier += 1
        self._length += 1
        pos = self._length - 1
        if len(self._items) <= pos:
            self._items.append(HeapNode(value, key, identifier))
        else:
            self._items[pos] = HeapNode(value, key, identifier)
        self._item_positions[identifier] = pos
        # While the new object has a parent and the parent is greater than the new object swap with parent.
        while pos != 0:
            v = self._items[self.parent_pos(pos)]
            if v.key > key:
                self.swap(pos, self.parent_pos(pos))
                pos = self.parent_pos(pos)
                continue
            else:
                break
        if self._validate:
            self.__check_valid()
        return identifier

    def decrease(self, identifier, new_key):
        """
        Decreases the element's key in O(log(N)) time
        :param identifier: The identifier of the element to modify.
        :param new_key: The new key value.
        """
        pos = self._item_positions[identifier]
        self._items[pos] = HeapNode(self._items[pos].value, new_key, identifier)
        # While parent is smaller than the new_key swap with parent.
        while True:
            v = self._items[self.parent_pos(pos)]
            if pos != 0 and v.key > new_key:
                self.swap(pos, self.parent_pos(pos))
                pos = self.parent_pos(pos)
                continue
            else:
                break
        if self._validate:
            self.__check_valid()
