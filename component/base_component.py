from abc import ABC, abstractmethod
from multiprocessing.managers import BaseManager, shared_memory
from multiprocessing import resource_tracker
import numpy as np
import argparse
import logging
import time
import signal

logging.basicConfig(level=logging.INFO)


class QueueManager(BaseManager): pass
QueueManager.register('get_queue')

data_dict = {
    'frame': {
        'dtype': 'u1',
        'shape': '(1080, 1920, 3)',
    },
    'bounding_boxes': {
        'dtype': 'f4',
        'shape': '(6,)',
        'length': 10000
    },
}

class BaseComponent(ABC):
    def __init__(self, config) -> None:
        self.manager = QueueManager(address=f'/dev/shm/pipeline_{config.pipeline_id}_manager', authkey=b'abc')

        # attempt to connect to pipeline manager with retry
        for i in range(config.init_retries):
            time.sleep(5)
            try:
                self.manager.connect()
            except FileNotFoundError:
                logging.info(f'Cannot connect to pipeline manager, retrying {i + 1} time(s)...')
                continue
            logging.info('Connected to pipeline manager')
            break

        service_queue = f'{config.service}_queue'
        next_service_queue = f'{config.next_service}_queue' if config.next_service else None

        self.queue = self.manager.get_queue(service_queue)
        self.next_queue = None
        if next_service_queue:
            self.next_queue = self.manager.get_queue(next_service_queue)

        event_dtype_list = []
        # parse data dict config
        for field, d_config in data_dict.items():
            dtype = np.dtype(f'{d_config["shape"]}{d_config["dtype"]}')
            if 'length' in d_config:
                # add len field
                len_field = f'_{field}_len'
                event_dtype_list.append((len_field, np.dtype('u4')))  # len has fixed dtype of uint32

                event_dtype_list.append((field, dtype, d_config['length']))
            else:
                event_dtype_list.append((field, dtype))
        event_dtype = np.dtype(event_dtype_list)

        self.d_shm = {}
        self.d_bufs = {}
        self.sl = shared_memory.ShareableList(name=f'pipeline_{config.pipeline_id}_uuids')
        for uuid in self.sl:
            self.d_shm[uuid] = shared_memory.SharedMemory(name=uuid.decode())
            self.d_bufs[uuid] = np.ndarray(shape=(1,), dtype=event_dtype, buffer=self.d_shm[uuid].buf)
            resource_tracker.unregister(self.d_shm[uuid]._name, "shared_memory")
        resource_tracker.unregister(self.sl.shm._name, "shared_memory")

    def shm_read(self, uuid, field):
        if 'length' in data_dict[field]:
            return self.d_bufs[uuid][field][0][:self.d_bufs[uuid][f'_{field}_len'][0]]
        else:
            return self.d_bufs[uuid][field][0]

    def shm_write(self, uuid, field, array):
        if 'length' in data_dict[field]:
            length = array.shape[0]

            self.d_bufs[uuid][f'_{field}_len'][:] = np.uint32(length)
            self.d_bufs[uuid][field][0][:length] = array
        else:
            self.d_bufs[uuid][field][0][:] = array

    def run(self):
        uuid = self.queue.get()
        logging.info(f'Processing frame from {uuid}...')

        # Read from shm
        input_names = ['frame']  # TODO: from config
        inputs = [self.shm_read(uuid, input_name) for input_name in input_names]

        # Run process function
        outputs = self.process(*inputs)

        # Write to shm
        output_names = ['frame']  # TODO: from config
        for i, output in enumerate(output_names):
            self.shm_write(uuid, output_names[i], output)

        if self.next_queue:
            self.next_queue.put(uuid)
        else:
            # recycle the uuid
            self.manager.get_queue('buffer_queue').put(uuid)

    @abstractmethod
    def process(self, *args, **kwargs):
        raise NotImplementedError

    def shutdown(self):
        self.sl.shm.close()

        for uuid, shm in self.d_shm.items():
            shm.close()
