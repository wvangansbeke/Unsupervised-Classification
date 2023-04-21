"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-100-python', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        if database == 'cifar-10':
            return './dataset/cifar-10/'

        elif database == 'cifar-100-python':
            return './dataset/cifar-100-python/'

        elif database == 'cifar-100':
            return './dataset/cifar-100/'

        elif database == 'MNIST':
            return './dataset/MNIST'

        elif database == 'stl-10':
            return './dataset/stl-10/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return './dataset/imagenet/'
        
        else:
            raise NotImplementedError
