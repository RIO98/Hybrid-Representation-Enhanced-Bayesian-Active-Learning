class DataLoaderIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = self.create_new_iterator()

    def create_new_iterator(self):
        return iter(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = self.create_new_iterator()
            batch = next(self.iterator)

        return batch
