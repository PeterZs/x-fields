
import tensorflow as tf
import numpy as np
import cv2,time,os
from blending import Blending_test
from load_imgs import load_imgs
from model import Flow
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, 
                    help='path to dataset',      )
parser.add_argument('--type',     type=str, nargs= "*",
                    help='xfield type',         )
parser.add_argument('--dim',      type=int, nargs= "*",
                    help='dimension of Xfields', )
parser.add_argument('--num_n',    type=int,
                    help='number of neighbors')
parser.add_argument('--factor',   type=int,
                    help='downsample factor')
parser.add_argument('--nfg',      type=int,
                    help='capacity multiplier')
parser.add_argument('--br',      type=float,
                    help='baseline ratio', default =1)
parser.add_argument('--sigma',    type=float,
                    help='exponential bandwidth', default =0.1)
parser.add_argument('--savepath', type=str,
                    help='saving path')
parser.add_argument('--scale',      type=float,
                    help='upsampling factor',  default = 60)
parser.add_argument('--fr',      type=float,
                    help='output video frame rate',  default = 60)

args = parser.parse_args()


def run_test(args):

    
    print('---------- Perform Testing ----------')

    savedir = args.savepath
    if not os.path.exists(savedir):
        raise NameError('There is no directory:\n %s'%(savedir))
    if not os.path.exists(os.path.join(savedir,"saved test") ):
        os.mkdir( os.path.join(savedir,"saved test") )
        print('creating directory %s'%(os.path.join(savedir,"saved test")))
    if not os.path.exists(os.path.join(savedir,"rendered videos") ):
        os.mkdir( os.path.join(savedir,"rendered videos") )
        print('creating directory %s'%(os.path.join(savedir,"rendered videos")))

    
    print('XField type: %s'%(args.type))
    print('Xfield dimensions: %s'%(args.dim))

    images,coordinates,all_pairs,h_res,w_res = load_imgs(args) 

  
    dims = args.dim   
    num_n = args.num_n
    savedir = args.savepath
    if num_n > np.prod(dims):
            num_n = np.prod(dims)
    
    input = tf.placeholder(tf.float32,shape=[1,1,1,len(dims)])    
    
    with tf.variable_scope("gen_flows"):
        flows = Flow(input,h_res,w_res,len(args.type)*2,args.nfg)

    if args.type == ['light','view','time']:

        with tf.variable_scope("gen_flows"):
                albedos = tf.Variable(tf.constant(1.0, shape=[dims[1]*dims[2],h_res, w_res,3]), name='albedo')
        index_albedo   = tf.placeholder(tf.int32,shape=(num_n,))
        albedo         = tf.gather(albedos,index_albedo,0)
       
    elif args.type == ['light']:
        
        with tf.variable_scope("gen_flows"):
            albedos = tf.Variable(tf.constant(1.0, shape=[dims[0]*dims[1],h_res, w_res,3]), name='albedo')   
            index_albedo = tf.placeholder(tf.int32,shape=(num_n,))
            albedo =   tf.gather(albedos,index_albedo,0)
            
    else:       
        albedo =   tf.constant(1.0, shape=[num_n,h_res, w_res,3]) 
       
    input_N        = tf.placeholder(tf.float32,shape=[num_n,1,1,len(dims)])
    Neighbors_img  = tf.placeholder(tf.float32,shape=[num_n,h_res,w_res,3])
    Neighbors_flow = tf.placeholder(tf.float32,shape=[num_n,h_res,w_res,len(args.type)*2])

    interpolated = Blending_test(input,input_N,Neighbors_img,Neighbors_flow,flows,albedo,h_res,w_res,args)
    
    saver = tf.train.Saver(max_to_keep=1000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    ckpt=tf.train.get_checkpoint_state("%s\\trained model\\"%(savedir))
    if ckpt:
        print('\n loading pretrained model  '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        raise NameError('There is no pretrained model located at dir:\n %s\\trained model\\'%(savedir))
        
    precomputed_flows = []
    
    for i in range(len(coordinates)):
        flows_out = sess.run(flows,feed_dict={input:coordinates[[i],::]})
        precomputed_flows.append(flows_out[0,::])
      
    precomputed_flows = np.stack(precomputed_flows,0)     
    


    if args.type == ['view']: 
        
        
        print('number of neighbors: %d'%(num_n))
        print('\n ---------  view interpolation ---------')
        
        
        theta = [np.pi/args.scale*i for i in range(args.scale+1)];
        X1 = 1 - np.cos(theta);
        X2 = 1 + np.cos(theta);
        Y1 = 1 + np.sqrt(1-(X1-1)**2)
        Y2 = 1 - np.sqrt(1-(X2-1)**2)
        
        X = np.append(X1,X2)
        Y = np.append(Y1,Y2)
        
        X = (args.dim[1]-1)*X/2
        Y = (args.dim[0]-1)*Y/2
            
        rendering_path = np.transpose([X,Y])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('%s/rendered videos/rendered_view.mp4'%(savedir),fourcc, args.fr, (w_res,h_res))
        for id in range(len(X)):
                
                input_coord = np.array([[[rendering_path[id,:]]]])
                indices = np.argsort(np.sum(np.square(input_coord[0,0,0,:]-coordinates[:,0,0,:]),-1))[:num_n]
              
              
                input_coord_N   = coordinates[indices,::]
                input_Neighbors = images[indices,::]
                input_flows     = precomputed_flows[indices,::]
        
                st = time.time()

                im_out = sess.run(interpolated,feed_dict={        input         :input_coord,
                                                                  input_N       :input_coord_N,
                                                                  Neighbors_img :input_Neighbors,
                                                                  Neighbors_flow:input_flows,
                                                                          })
                im_out = np.minimum(np.maximum(im_out[0,::],0.0),1.0)
                out.write(np.uint8(im_out*255))
            
                print('\r interpolated image %d of %d  fps: %2.2f '%(id+1,len(rendering_path),1/(time.time()-st)),end=" " )

    
        out.release()
    
       
    if args.type == ['time']: 
        
            
        print('number of neighbors: %d'%(num_n))
        X = np.linspace(0,dims[0]-1,(dims[0]-1)*args.scale+dims[0])
        rendering_path = np.append(X,np.flip(X))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('%s/rendered videos/rendered_time.mp4'%(savedir),fourcc, args.fr, (w_res,h_res))

        print('\n---------  time interpolation ---------')

        for id in range(len(rendering_path)):
                
                input_coord = np.array([[[[rendering_path[id]]]]])
                indices = np.argsort(np.abs(input_coord[0,0,0,0]-coordinates[:,0,0,0]))[:num_n]

              
                input_coord_N   = coordinates[indices,::]
                input_Neighbors = images[indices,::]
                input_flows     = precomputed_flows[indices,::]
                
                st = time.time()

                im_out = sess.run(interpolated,feed_dict={        input         :input_coord,
                                                                  input_N       :input_coord_N,
                                                                  Neighbors_img :input_Neighbors,
                                                                  Neighbors_flow:input_flows,
                                                                          })
                im_out = np.minimum(np.maximum(im_out[0,::],0.0),1.0)
                out.write(np.uint8(im_out*255))
            
                print('\r interpolated image %d of %d  fps: %2.2f '%(id+1,len(rendering_path),1/(time.time()-st)),end=" " )
    
    
        out.release()
    
    if args.type == ['light','view','time']:
          
        print('\n number of neighbors for interpolation: %d'%(num_n))
        X_L = np.linspace(0,dims[0]-1,(dims[0]-1)*args.scale+dims[0])
        X_L = np.append(X_L,np.flip(X_L))
        X_V = np.linspace(0,dims[1]-1,(dims[1]-1)*args.scale+dims[1])
        X_V = np.append(X_V,np.flip(X_V))
        X_T = np.linspace(0,dims[2]-1,(dims[2]-1)*args.scale+dims[2])
        X_T = np.append(X_T,np.flip(X_T))        
               
        all_dimensions = {'light' : np.stack([X_L,np.ones_like(X_V)*dims[1]//2,np.ones_like(X_T)*dims[2]//2],1),
                          'view'   : np.stack([np.ones_like(X_L)*dims[0]//2,X_V,np.ones_like(X_T)*dims[2]//2],1),
                          'time'   : np.stack([np.ones_like(X_L)*dims[0]//2,np.ones_like(X_V)*dims[1]//2,X_T],1),
                          'light_view' :np.stack([X_L,X_V,np.ones_like(X_T)*dims[2]//2],1),
                          'light_time' :np.stack([X_L,np.ones_like(X_V)*dims[1]//2,X_T],1),
                          'view_time'  : np.stack([np.ones_like(X_L)*dims[0]//2,X_V,X_T],1),
                          'light_view_time' : np.stack([X_L,X_V,X_T],1) }
  
     
        for case, rendering_path in all_dimensions.items():

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('%s/rendered videos/rendered_%s.mp4'%(savedir,case),fourcc, args.fr, (w_res,h_res))
            
            st = time.time()
            print('\n --------- %s interpolation ---------'%(case))

            for id in range(len(rendering_path)):
                                       
                input_coord = np.array([[[rendering_path[id,:]]]])
                indices = np.argsort(np.sum(np.square(input_coord[0,0,0,:]-coordinates[:,0,0,:]),-1))[:num_n]
                            
                input_coord_N   = coordinates[indices,::]
                input_Neighbors = images[indices,::]
                input_flows     = precomputed_flows[indices,::]
                albedo_index    = input_coord_N[:,0,0,1]*dims[1] + input_coord_N[:,0,0,2]
        
                st = time.time()

                im_out = sess.run(interpolated,feed_dict={ input         :input_coord,
                                                           input_N       :input_coord_N,
                                                           Neighbors_img :input_Neighbors,
                                                           Neighbors_flow:input_flows,
                                                           index_albedo  :albedo_index,
                                                                          })
               
                im_out = np.minimum(np.maximum(im_out[0,::],0.0),1.0)
                out.write(np.uint8(im_out*255))
                
                print('\r interpolated image %d of %d  fps: %2.2f '%(id+1,len(rendering_path),1/(time.time()-st)),end=" " )
        
            out.release()
            
            

if __name__=='__main__':
       
    run_test(args)


