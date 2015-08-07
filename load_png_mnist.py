import numpy
#import pylab
from PIL import Image
import cPickle
import gzip
import os
import sys
import time
import random

import theano
import theano.tensor as T

import math
from glob    import glob
from os.path import join
from os.path import expanduser
import getpass
#import pickle_local as plocal
#import mnist_loader as mnistl
#from enum_local import LOAD

def get_dataset(alphabet_set = False, 
        test_only = False, 
        img_pickle_ld = True, 
        img_pickle_sv = True, 
        nist_stretch = 2,
        large_size = -1 ,
        save_filename = "",
        randomize = False) :
        
    epochs = 10
    if (save_filename != "") :
        nist_stretch, large_size, epochs, save_filename , load , load_type = parse_filename(save_filename)
        if load :
            img_pickle_ld = True
        else :
            img_pickle_sv = False
        
    nist_num = nist_stretch  *  1024 * 5 ## TOTAL IMG SPACE REQUESTED (images)
    if (large_size == -1) :
        large_size = 1024 * 10 * (28 * 28)  ## LIMIT SIZE (not images)
    else:
        size = 28, 28
        img2 = numpy.zeros(shape=(size), dtype='float32')
        s = sys.getsizeof(img2)
        print(str(s) + " size of 28x28")
        if (large_size > s) : large_size = large_size /  (s) ## (convert to num of images.)
        else : print ("small 'BigBatch' size, using as num of images.")
    rval_array = []
    pickle_len = 0
    obj = [[],0]
    dset = [[obj,obj,obj]]
    print("images requested: " + str(nist_num))

    t1 = []
    l1 = []
    t2 = []
    l2 = []
    t3 = []
    l3 = []

    if img_pickle_ld and (not img_pickle_sv) and load_type != LOAD.NUMERIC :
        dset = plocal.load_pickle_img(save_filename);
        #for j in dset :
        pickle_len =  len(dset[0][0][0] )
        
        pickle_len = int(pickle_len  )
        print(pickle_len, nist_num)
        if pickle_len < nist_num : nist_num = pickle_len
    
    if nist_num > large_size:
        print("Automatically splitting input files.")
        m = int(math.ceil(nist_num / float(large_size)))
    else:
        m = 1
    if  (not img_pickle_sv) :
        
        for zz in range(m):
            print('get dataset. ' + str(zz+1) + ' of ' + str(m))
            
            if (not alphabet_set) or load_type == LOAD.NUMERIC   :
                t1,l1, files = batch_load("normpic" , 1, 60000, randomize=randomize); ## 60000
                t2,l2, files = batch_load("normvalid", 1, 5000, randomize = randomize);
                t3,l3 ,files = batch_load("normtest", 1,  5000, randomize = randomize);
                print "load numeric"
            if alphabet_set and (not load_type == LOAD.NUMERIC) :
                
                stretch =  int(nist_num / float(10)) 
                
                if nist_num > large_size :
                    stretch = int(large_size / float(10) )
                    #print('middle')
                
                if nist_num > large_size and zz == m -1 : 
                    stretch = int (((nist_num ) - (zz * large_size  ))/ float(10))  
                    #print('end')
                
                if  True : ##not img_pickle_sv : ## not nist_num == pickle_len and
                
                    train_start = zz * large_size; # start with zero
                    train_stop =  train_start + (stretch * 8);
                    valid_start = train_stop + 1;
                    valid_stop =  valid_start + (stretch * 1);
                    test_start =  valid_stop + 1;
                    test_stop =   test_start + (stretch * 1);

                    
                    if (not img_pickle_ld) and load_type != LOAD.NUMERIC :
                        if not test_only :
                            files = []
                            t1, l1, files = batch_load_alpha( train_start,  train_stop, randomize, files, load_type)
                            t2, l2, files = batch_load_alpha( valid_start,  valid_stop, randomize, files, load_type)
                            
                        else:
                            t1 = l1 = t2 = l2 = files = []
                            
                        t3, l3 , files = batch_load_alpha( test_start,   test_stop, randomize, files, load_type)
                    
                    
                    elif img_pickle_ld :
                        print('Loading pickle file: ~/workspace/nn/' + save_filename )
                        print('delete this file to disable loading.')
                        
                        dset2 = dset[0][0]
                        #print dset2[1]
                        
                        t1, l1 = dset2[0][train_start:train_stop] , dset2[1][train_start:train_stop]
                        t2, l2 = dset2[0][valid_start:valid_stop] , dset2[1][valid_start:valid_stop]
                        t3, l3 = dset2[0][test_start:test_stop]   , dset2[1][test_start:test_stop]
                        print('train: ' + str(len(l1)))
                        print('valid: ' + str(len(l2)))
                        print('test:  ' + str(len(l3)))
                        
                    
            rval = [(t1, l1), (t2, l2), (t3, l3)]


            rval_array.append( rval )
        
    elif (not test_only)  and img_pickle_sv :
        num_start = 0
        num_stop = nist_num
        t1, l1 , files = batch_load_alpha( num_start,  num_stop, randomize=False, files = [], load_type=load_type)
        print('train: ' + str(len(l1)))
        
        #plocal.save_pickle_img( rval_array ,[],[], filename = save_filename );
        plocal.save_pickle_img(  [[[t1, l1]]]  ,[],[], filename = save_filename );
        print('\nImage Pickle Save: only works for small image sets!')
        print('(it hogs memory and will freeze your computer.)')
    return rval_array , epochs, nist_stretch, load_type

	
def batch_load(basename = "normpic",
        series_start = 1, 
        series_stop = 1, 
        foldername = "../oldpng-mnist/",
        files = [],
        randomize = False):
        
    train_set = []
    train_num = []


    mnist = mnistl.MNIST(foldername)
    if basename == "normpic" :
        train_set, train_num = mnist.load_training()
    if basename == "normtest" :
        train_set, train_num = mnist.load_conv_test()
    if basename == "normvalid" :
        train_set, train_num = mnist.load_conv_valid()

    print "len of mnist sets " + str(len(train_num))
    if randomize :
        out_img = []
        out_num = []

        for ii in range(series_stop - series_start ) :

            k = random.randint(0, len(train_num) - 1)

            if True :
                for i in range(0, 28*28) :
                    if train_set[k][i] > 100  :
                        train_set[k][i] = 1
                    else :
                        train_set[k][i] = 0

            out_img.append(train_set[k])
            out_num.append(train_num[k])

            del train_set[k]
            del train_num[k]

        new_list = []
        if False :
            for jj in out_img:
                if isinstance(jj, list) :
                    new_list.append(jj)
            print("----")
            print(out_img)
            print(len(out_img))

        return out_img, out_num, files


    return train_set, train_num, files
	
def filename_list(series_start=0, series_stop=2, randomize = False , files = [] , load_type = 0):
    
    if len(files) == 0 :
        files = []
        folder = 'F*'
        #folder_username = getpass.getuser()
        home = expanduser("~")
        
        #print(folder)
        g = glob(join(home ,'workspace','sd_nineteen','HSF_0',folder))
        h = glob(join(home ,'workspace','sd_nineteen','HSF_1',folder))
        g.extend(h)
        i = glob(join(home ,'workspace','sd_nineteen','HSF_2',folder))
        g.extend(i)
        jj = glob(join(home ,'workspace','sd_nineteen','HSF_3',folder))
        g.extend(jj)
        kk = glob(join(home ,'workspace','sd_nineteen','HSF_4',folder))
        g.extend(kk)
        ll = glob(join(home ,'workspace','sd_nineteen','HSF_6',folder))
        g.extend(ll)
        mm = glob(join(home ,'workspace','sd_nineteen','HSF_7',folder))
        g.extend(mm)
        g.sort()
        
        #print ("sorted folder list: ", len(g))
        for j in g : #range(series_start, series_stop):
            gg = glob(join( j ,'*.bmp'))
            #print ("list: ",gg)
            files.extend(gg)
        print ('loadable files: '+ str(len(files)))
        print ('loaded files  : ' + str(int(series_stop - series_start)))
        files.sort()
    output = []
    if not randomize :
        output = files[int(series_start): int(series_stop) ]
    else :
        print len(files)
        num_files = int( series_stop - series_start )
        for j in range(num_files) :
            digit_start = 48
            k = random.randint(0, len(files))
            xxx, d = get_number(files[k], load_type)

            while d >= digit_start and d < digit_start + 10 and load_type == LOAD.ALPHA :
                #print d - digit_start
                del files[k]
                k = random.randint(0, len(files))
                xxx, d = get_number(files[k], load_type)

            xxx, d = get_number(files[k], load_type)
            if d >= digit_start and d < digit_start + 10 and load_type == LOAD.ALPHA:
                print "file error number detected"
                exit()
            #if j is 0 : print files[k]
            output.append(files[k])
            del files[k]
            
    return output, files
    
	
def batch_load_alpha(series_start = 1, series_stop = 1, randomize = False, files = [], load_type =0):
    img_list , files = filename_list(series_start, series_stop, randomize, files, load_type )
    train_set = []
    train_num = []
    oneimg = []
    oneindex = 0
    i = 0
    if (len(img_list) > 0) and True:
        print('sample: ' + img_list[0])
        sys.stdout.flush()
    
    for filename in img_list:

        oneimg, oneindex = look_at_img(filename, load_type=load_type)
        train_set.append(oneimg)
        train_num.append(oneindex)
        #print(filename)
    return train_set, train_num, files

def look_at_img( filename , i = 0, load_type =0):
    img = Image.open(open( filename ))
    size = 28, 28
    img2 = numpy.zeros(shape=(size), dtype='float64')
    oneimg = []
    oneindex = i
    xy_list = []
    
    img = numpy.asarray(img, dtype='float64')
    marker = 0
    ''' Detect 0 for black -- put in list in shrunk form. '''
    for x in range(0,len(img)):
        for y in range(0, len(img)):
            if (float(img[x,y,0]) < 255) is  True:
                xy_list.append([x* 1/float(2) - 18,y * 1/float(2) - 18])
    
    ''' Put list in 28 x 28 array. '''
    if len(xy_list) == 0 :
        xy_list = [0,0]
    for q in xy_list :
        if (q[0] < 28) and (q[1] < 28) and (q[0] >= 0) and (q[1] >= 0):
            #print (q[0], q[1])
            img2[int(math.floor(q[0])), int(math.floor(q[1]))] = 1
    
    ''' Then add entire array to oneimg variable and flatten.'''
    for x in range(28) :
        for y in range(28) :
            oneimg.append(img2[x,y])
    
    ''' Get the image ascii number from the filename. '''
    oneindex , unused = get_number(filename, load_type)
    return oneimg, oneindex

def get_number(filename, load_type ):
    mat = ascii_matrix(load_type)
    newindex = 0
    index = 0
    l_bmp = len('.bmp')  ## discard this many chars for ending
    l_sample = l_bmp + 2 ## sample two chars
    
    l_filename = len(filename)
    filename = filename[l_filename - l_sample : l_filename - l_bmp] ## slice
    if filename[0:1] is '_':
        filename = filename[1: len(filename)] ## slice again
        ## consume first char.
    filename = '0x' + filename
    index = int(filename, 16) ## translate hex to int
    for i in range(len(mat)) :
        if index is mat[i][0] :
            newindex = i
    return newindex, index

def ascii_matrix(alphabet_set ) :
    mat = []
    a_upper = 65 ## ascii for 'A'
    a_lower = 97 ## ascii for 'a'
    z_digit = 48 ## ascii for '0'
    
    if alphabet_set == LOAD.ALPHANUMERIC or alphabet_set == LOAD.ALPHA :
        for i in range(0,26):
            value = int(a_upper + i) , str(unichr(a_upper+i))
            mat.append(value)
        for i in range(0,26):
            value = int(a_lower + i) , str(unichr(a_lower+i))
            mat.append(value)

    if alphabet_set == LOAD.ALPHANUMERIC or alphabet_set == LOAD.NUMERIC : ## do not seperate nums and alphabet yet.
        for i in range(0,10):
            value = int(z_digit + i) , str(unichr(z_digit+i))
            mat.append(value)
    if len(mat) == 0 :
        print ("load type " + str(alphabet_set)), LOAD.ALPHA , LOAD.NUMERIC, LOAD.ALPHANUMERIC
        raise RuntimeError
    #print(len(mat))
    return mat


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
        dtype=theano.config.floatX),
        borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
        dtype=theano.config.floatX), 
        borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def show_xvalues(xarray = [[]], index = 0):
    print ("show x values " + str(index))
    xx = xarray[index]
    ln = int(math.floor(math.sqrt(len(xx)))) 
    #print (ln)
    for x in range(1,ln):
        for y in range(1, ln):
            zzz = '#'
            #zzz =int( xx[x* ln + y])
            if int(xx[ x* ln + y]) == int( 0) : 
                zzz = '.'
            #print(zzz) 
            sys.stdout.write(zzz)
        print("");
    print ("\n===========================\n")

def parse_filename(filename=""):
    nist_stretch = 2
    large_size = 1
    epochs = 10
    split_filename = filename.split("/")
    save_filename = split_filename[len(split_filename) - 1]
    tag1 = "save"
    tag2 = "x5K-images"
    tag3 = "GB-limit"
    tag4 = "epochs.save"
    tag5 = "run" ## IMPLIED ALPHA-NUMERIC!!
    tag6 = "alpha"
    tag7 = "numeric"
    s = save_filename.split("_")
    #print(s)
    load = False

    load_type = 0
    g = s[0].split("-")
    s[0] = g[0]
    if len(g) == 1 :
        load_type = LOAD.ALPHANUMERIC
        print("load both")
    elif g[1] == tag6 :
        load_type = LOAD.ALPHA
        print("load alpha")
    elif g[1] == tag7 :
        load_type = LOAD.NUMERIC
        print("load numeric")

    if ( s[0] == tag1 and s[2] == tag2 and s[4] == tag3 and s[6] == tag4) :
        nist_stretch = int(s[1].strip())
        large_size = int(float(s[3].strip()) * float(math.pow(2,30)) ) #1000000000
        epochs = int(s[5].strip())
        load = True
    elif (s[0] == tag5 and s[2] == tag2 and s[4] == tag3 and s[6] == tag4) :
        nist_stretch = int(s[1].strip())
        large_size = int(float(s[3].strip()) * float(math.pow(2,30)) ) #1000000000
        epochs = int(s[5].strip())
    else :
        print("Poorly formed file name!")
        exit();
    return nist_stretch, large_size, epochs, save_filename, load , load_type;
    

