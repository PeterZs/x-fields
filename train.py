
import numpy as np
import cv2,time,os
import flow_vis
from blending import Blending_train
from load_imgs import load_imgs
from model import Flow
import argparse
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, 
                    help='path to dataset',      default = 'dataset\\light_view_time\\chair')
parser.add_argument('--type',     type=str, nargs= "*",
                    help='xfield type',          default = ['light','view','time'])
parser.add_argument('--dim',      type=int, nargs= "*",
                    help='dimension of Xfields', default = [5,5,5])
parser.add_argument('--factor',   type=int,
                    help='downsample factor',    default = 6)
parser.add_argument('--nfg',      type=int,
                    help='capacity multiplier',  default = 8)
parser.add_argument('--num_n',    type=int,
                    help='number of neighbors',  default = 2)
parser.add_argument('--lr',       type=float,
                    help='learning rate',        default = 0.0001)
parser.add_argument('--sigma',    type=float,
                    help='bandwidth parameter',default = 0.1)
parser.add_argument('--br',      type=float,
                    help='baseline ratio',       default = 1)
parser.add_argument('--load_pretrained', type=bool,
                    help='loading pretrained model',default = False)
parser.add_argument('--savepath', type=str,
                    help='saving path',          default = 'outputs\\chair')

args = parser.parse_args()


def run_training(args):

    print('---------- Perform Training ----------')
    
    savedir = args.savepath
    print('saving directory: %s'%(savedir))
    if not os.path.exists(savedir):
        os.mkdir( savedir )
    if not os.path.exists(os.path.join(savedir,"trained model") ):
        os.mkdir( os.path.join(savedir,"trained model") )
        print('creating directory %s'%(os.path.join(savedir,"trained model")))
    if not os.path.exists(os.path.join(savedir,"saved training") ):
        os.mkdir( os.path.join(savedir,"saved training") )
        print('creating directory %s'%(os.path.join(savedir,"saved training")))

    
    
    print('XField type: %s'%(args.type))
    print('Xfield dimensions: %s'%(args.dim))
    
    images,coordinates,all_pairs,h_res,w_res = load_imgs(args) 

    dims = args.dim
    num_n = args.num_n
    savedir = args.savepath
    
    inputs = tf.placeholder(tf.float32,shape=[num_n+1,1,1,len(dims)])    

    print('\n ------- Creating the model -------')
    
    with tf.variable_scope("gen_flows"):
        flows = Flow(inputs,h_res,w_res,len(args.type)*2,args.nfg)

    nparams_decoder = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if v.name.startswith("gen_flows")])
    print('Number of learnable parameters (decoder): %d' %(nparams_decoder))

    if args.type == ['light','view','time']:
        with tf.variable_scope("gen_flows"):
            albedos = tf.Variable(tf.constant(1.0, shape=[dims[1]*dims[2],h_res, w_res,3]), name='albedo')   
            index_albedo = tf.placeholder(tf.int32,shape=(1,))
            albedo =   tf.gather(albedos,index_albedo,0)
        nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if v.name.startswith("gen_flows")])
        print('Number of learnable parameters (%d albedos with res %d x %d ): %d' %(dims[1]*dims[2],h_res,w_res,nparams-nparams_decoder))
    
    elif args.type == ['light']:
        with tf.variable_scope("gen_flows"):
            albedos = tf.Variable(tf.constant(1.0, shape=[dims[0]*dims[1],h_res, w_res,3]), name='albedo')   
            index_albedo = tf.placeholder(tf.int32,shape=(1,))
            albedo =   tf.gather(albedos,index_albedo,0)
        nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if v.name.startswith("gen_flows")])
        print('Number of learnable parameters (%d albedos with res %d x %d ): %d' %(dims[0]*dims[1],h_res,w_res,nparams-nparams_decoder))
    
    else:
        albedo =   tf.constant(1.0, shape=[1,h_res, w_res,3])

    Neighbors = tf.placeholder(tf.float32,shape=[num_n,h_res,w_res,3])
    interpolated = Blending_train(inputs,Neighbors,flows,albedo,h_res,w_res,args)
    
    Reference = tf.placeholder(tf.float32,shape=[1,h_res,w_res,3])    
    loss  = tf.reduce_mean((tf.abs(interpolated-Reference)))
   
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("gen_flows")]
    learning_rate = tf.placeholder(tf.float32,shape=())
    gen_optim = tf.train.AdamOptimizer(learning_rate)
    gen_grads = gen_optim.compute_gradients(loss, var_list=gen_tvars)
    gen_train = gen_optim.apply_gradients(gen_grads)
    
    saver = tf.train.Saver(max_to_keep = 1000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    if args.load_pretrained:
        ckpt=tf.train.get_checkpoint_state("%s\\trained model"%(savedir))
        if ckpt:
                print('\n loading pretrained model  '+ckpt.model_checkpoint_path)
                saver.restore(sess,ckpt.model_checkpoint_path)

    print('------------ Start Training ------------')
    
    lr = args.lr
    indices = np.array([i for i in range(len(all_pairs))])
#    iter_num = 4000*np.prod(dims)//args.nfg

    if len(indices)<100:
      indices = np.repeat(indices,100//len(indices))
        
    epoch_size = len(indices)
#    epoch_num = iter_num//epoch_size

#    print('Number of epochs: %d'%(epoch_num))
#    print('Epoch size: %d'%(epoch_size))
#    print('Total number of iterations: %d'%(epoch_num*epoch_size))
#    print('learning rate: %0.4f'%(lr))
    
    min_loss = 1000
    l1_loss_t = 1
    stop_l1_thr = 0.011
    epoch = 0
    if args.type == ['light','view','time']:
    
        st = time.time()

        while l1_loss_t > stop_l1_thr:
                
               l1_loss_t =0
               np.random.shuffle(indices)
              
              
#               if epoch == 3*epoch_num//4:
#                 lr =  args.lr /10
#                 print('\n decreasing learning rate to %0.5f'%(lr))
                         
               for id in range(epoch_size):                    
        
                   pair =  all_pairs[indices[id],::]
    
                   input_coords = coordinates[pair[:num_n+1],::]
                   reference_img = images[pair[:1],::]
                   Neighbors_img = images[pair[1:num_n+1],::]
                   _index = [pair[-1]]
        
                   _,l1loss = sess.run([gen_train,loss],feed_dict={inputs:input_coords,
                                                                   Reference:reference_img,
                                                                   Neighbors: Neighbors_img,
                                                                   learning_rate:lr,
                                                                   index_albedo:_index
                                                                   })
                   l1_loss_t = l1_loss_t + l1loss
    
                   print('\r Epoch %3.0d  Iteration %3.0d of %3.0d   Cumulative L1 loss = %3.3f'%(epoch,id+1,epoch_size,l1_loss_t),end=" " )
        
                                            
               l1_loss_t = l1_loss_t/epoch_size
               print(" elapsed time %3.1f m  Averaged L1 loss = %3.5f "%((time.time()-st)/60,l1_loss_t))
        
               if l1_loss_t < min_loss:
                      saver.save(sess,"%s\\trained model\\model.ckpt"%(savedir))
                      min_loss = l1_loss_t
               
               center = np.prod(dims)//2 
               cv2.imwrite("%s/saved training/reference.png"%(savedir),np.uint8(images[center,::]*255))
   
    
               pair =  all_pairs[3*center + 0,::]
    
               out_img,flows_out = sess.run([interpolated,flows],feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                                            Neighbors   :images[pair[1:num_n+1],::],
                                                                            index_albedo:[pair[-1]]})
                                                                            
               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved training/recons_light.png"%(savedir),np.uint8(out_img[0,::]*255))
        
               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,0:2], convert_to_bgr=False)
               cv2.imwrite("%s/saved training/flow_light.png"%(savedir),np.uint8(flow_color))
               
               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,2:4], convert_to_bgr=False)
               cv2.imwrite("%s/saved training/flow_view.png"%(savedir),np.uint8(flow_color))
          
               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,4:6], convert_to_bgr=False)
               cv2.imwrite("%s/saved training/flow_time.png"%(savedir),np.uint8(flow_color))
    
                   
               pair =  all_pairs[3*center + 1,::]  
               out_img = sess.run(interpolated,feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                          Neighbors   :images[pair[1:num_n+1],::],
                                                          index_albedo:[pair[-1]]})
                  
               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved training/recons_view.png"%(savedir),np.uint8(out_img[0,::]*255))
        
        
               pair =  all_pairs[3*center + 2,::]
               out_img = sess.run(interpolated,feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                          Neighbors   :images[pair[1:num_n+1],::],
                                                          index_albedo:[pair[-1]]})
                                                                                                                                                   
               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved training/recons_time.png"%(savedir),np.uint8(out_img[0,::]*255))
               epoch  = epoch + 1
        
    if args.type == ['view'] or args.type ==['time']:
        
        
        st = time.time()
        img_mov = cv2.VideoWriter('%s/saved training/epoch_recons.mp4'%(savedir),cv2.VideoWriter_fourcc(*'mp4v'), 10, (w_res,h_res))
        flow_mov = cv2.VideoWriter('%s/saved training/epoch_flows.mp4'%(savedir),cv2.VideoWriter_fourcc(*'mp4v'), 10, (w_res,h_res))

        while l1_loss_t > stop_l1_thr :
                
               l1_loss_t = 0
               np.random.shuffle(indices)
                           
#               if epoch == 3*epoch_num/4:
#                 lr =  args.lr /10
#                 print('\n decreasing learning rate to %0.5f'%(lr))
#                         
                      
               for id in range(epoch_size):                    
        
                   pair          =  all_pairs[indices[id],::]
                   input_coords  = coordinates[pair[:num_n+1],::]
                   reference_img = images[pair[:1],::]
                   Neighbors_img = images[pair[1:num_n+1],::]
        
                   _,l1loss = sess.run([gen_train,loss],feed_dict={inputs:input_coords,
                                                                   Reference:reference_img,
                                                                   Neighbors: Neighbors_img,
                                                                   learning_rate:lr,
                                                                   })
    
                   l1_loss_t = l1_loss_t + l1loss
                   print('\r Epoch %3.0d  Iteration %3.0d of %3.0d   Cumulative L1 loss = %3.3f'%(epoch,id+1,epoch_size,l1_loss_t),end=" " )
        
                                            
               l1_loss_t = l1_loss_t/epoch_size
               print(" elapsed time %3.1f m  Averaged L1 loss = %3.5f "%((time.time()-st)/60,l1_loss_t))
        
               if l1_loss_t < min_loss:
                      saver.save(sess,"%s\\trained model\\model.ckpt"%(savedir))
                      min_loss = l1_loss_t
        
            
               center = np.prod(dims)//2 
               cv2.imwrite("%s/saved training/reference.png"%(savedir),np.uint8(images[center,::]*255))
       
               pair =  all_pairs[(len(all_pairs)//len(images)) *center,::]
    
               out_img,flows_out = sess.run([interpolated,flows],feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                                            Neighbors   :images[pair[1:num_n+1],::]})
                                                                            
               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved training/recons.png"%(savedir),np.uint8(out_img[0,::]*255))
        
               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,0:2], convert_to_bgr=False)
               cv2.imwrite("%s/saved training/flow.png"%(savedir),np.uint8(flow_color))
               img_mov.write(np.uint8(out_img[0,::]*255))
               flow_mov.write(np.uint8(flow_color))
               epoch  = epoch + 1
        

        img_mov.release()
        flow_mov.release()
       
if __name__=='__main__':
    
 
    run_training(args)
