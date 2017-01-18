from __future__ import absolute_import
from __future__ import unicode_literals
#
# (c) Bernhard X. Kausler, 2010
#
from . import conversion as c
import unittest as ut
import os.path as path
import shutil
import tempfile

dataDir = "test_data/conversion/"

allZerosMat = dataDir + "all_zeros_stack.mat"
allZerosHdf5 = dataDir + "all_zeros_stack.h5"

maxIntMat = dataDir + "max_int_stack.mat"
maxIntHdf5 = dataDir + "max_int_stack.h5"

randomMat = dataDir + "random_stack.mat"
randomHdf5 = dataDir + "random_stack.h5"



class TestCompare( ut.TestCase ):
    def test_sameMat(self):
        self.assertTrue( c.compare(allZerosMat, allZerosMat) )
        self.assertTrue( c.compare(maxIntMat, maxIntMat) )
        self.assertTrue( c.compare(randomMat, randomMat) )
    
    def test_sameHdf5(self):
        self.assertTrue( c.compare(allZerosHdf5, allZerosHdf5) )
        self.assertTrue( c.compare(maxIntHdf5, maxIntHdf5) )
        self.assertTrue( c.compare(randomHdf5, randomHdf5) )
 
    def test_MatToHdf5(self):
        self.assertTrue( c.compare(allZerosMat, allZerosHdf5) )
        self.assertTrue( c.compare(maxIntMat, maxIntHdf5) )
        self.assertTrue( c.compare(randomMat, randomHdf5) )
 
    def test_Hdf5ToMat(self):
        self.assertTrue( c.compare(allZerosHdf5, allZerosMat) )
        self.assertTrue( c.compare(maxIntHdf5, maxIntMat) )
        self.assertTrue( c.compare(randomHdf5, randomMat) )
 


class TestConvert( ut.TestCase ):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="empryonic")
        
        self.allZerosHdf5 = path.join(self.tempdir, "all_zeros.h5")
        self.maxIntHdf5 = path.join(self.tempdir, "max_int.h5")
        self.randomHdf5 = path.join(self.tempdir, "random.h5")
        
    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_convertWithoutCompr(self):
        """Compare converted data to original data and a manual conversion."""
        c.convert(allZerosMat, self.allZerosHdf5)
        self.assertTrue( c.compare(allZerosMat, self.allZerosHdf5) )
        self.assertTrue( c.compare(self.allZerosHdf5, allZerosHdf5) )

        c.convert(maxIntMat, self.maxIntHdf5)
        self.assertTrue( c.compare(maxIntMat, self.maxIntHdf5) )
        self.assertTrue( c.compare(self.maxIntHdf5, maxIntHdf5) )

        c.convert(randomMat, self.randomHdf5)
        self.assertTrue( c.compare(randomMat, self.randomHdf5) )
        self.assertTrue( c.compare(self.randomHdf5, randomHdf5) )

    def test_convertWithCompr(self):
        """Compare converted data to original data and a manual conversion."""
        c.convert(allZerosMat, self.allZerosHdf5, compr=4)
        self.assertTrue( c.compare(allZerosMat, self.allZerosHdf5) )
        self.assertTrue( c.compare(self.allZerosHdf5, allZerosHdf5) )

        c.convert(maxIntMat, self.maxIntHdf5, compr=2)
        self.assertTrue( c.compare(maxIntMat, self.maxIntHdf5) )
        self.assertTrue( c.compare(self.maxIntHdf5, maxIntHdf5) )

        c.convert(randomMat, self.randomHdf5, compr=9)
        self.assertTrue( c.compare(randomMat, self.randomHdf5) )
        self.assertTrue( c.compare(self.randomHdf5, randomHdf5) )



class TestBatchConvert( ut.TestCase ):
    def setUp( self ):
        self.tempdir = tempfile.mkdtemp(prefix="empryonic")

        shutil.copy(allZerosMat, self.tempdir)
        shutil.copy(maxIntMat, self.tempdir)
        shutil.copy(randomMat, self.tempdir)

    def tearDown( self ):
        shutil.rmtree(self.tempdir)

    def runTest( self ):
        c.batchConvert( self.tempdir, compression=4)
        self.assertTrue( c.compare(path.join(self.tempdir, "all_zeros_stack.mat"),
                                   path.join(self.tempdir, "all_zeros_stack.h5")) )
        self.assertTrue( c.compare(path.join(self.tempdir, "max_int_stack.mat"),
                                   path.join(self.tempdir, "max_int_stack.h5")) )
        self.assertTrue( c.compare(path.join(self.tempdir, "random_stack.mat"),
                                   path.join(self.tempdir, "random_stack.h5")) )



if __name__ == "__main__":
    ut.main()
