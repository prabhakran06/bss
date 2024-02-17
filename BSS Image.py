##PROJECT BSS: IMAGE

##importing modules(run successfully)
from PIL import Image
import numpy as np
import cv2
import math
import random
import sympy

#======================================================================
##PROBLEM:    1.ACCESSING THE CAMERA AND SAVE THE IMAGE.
##STATUS:      (runs successfully)

def capture_image():
    #capturing the image
    print("See the camera and Smile please....")
    # create a video capture object
    cap = cv2.VideoCapture(0)
    l=0
    for i in range(100):
        for j in range(100):
            for k in range(100):
                l=l+2
    # check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video capture")
    
    # capture an image
    ret, frame = cap.read()
    
    # save the image to a file
    cv2.imwrite('Original.jpg', frame)
    
    # release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
    print("\t\t\t!!!  Image has been saved  !!!")

#======================================================================

##PROBLEM:      CONVERTING THE  IMAGE INTO ARRAY(FLOAT VALUES)
##STATUS:Runs Successfully

def get_image(image):
    # Open the image with PIL
    img = Image.open(image)
    
    # Convert the PIL image to a NumPy array
    img_array = np.asarray(img)
    
    # Convert the NumPy array to float values
    img_float = img_array.astype(np.float32) / 1.0
    
    return img_float
#======================================================================

##PROBLEM:      CONVERTING THE ARRAY INTO IMAGE(FLOAT VALUES)
##STATUS:Runs Successfully

def set_image(file_name,img_float):
    # Convert the NumPy array back to an image
    img_float = (img_float * 1.0).astype(np.uint8)
    img = Image.fromarray(img_float)
    img.save(file_name)
#======================================================================

##PROBLEM:    GETTING THE SIZE
##STATUS:Runs Successfully

def get_size(image):
    # Open the image with PIL
    img = Image.open(image)

    #Get the size of the image
    width, height = img.size
    print("Image has been loaded and converted into the matrix!!!!")
    
    return width, height

#======================================================================
##PROBLEM:    GETTING THE ONE PIXEL CHANGED IMAGE
##STATUS:Runs Successfully

def get_new_image(height,width,img_float_I2,key):

    key=key%9
    for i in range(height):
        for j in range(width):
            k=random.randint(0,2)
            log=key+(k*k)
            img_float_I2[i][j][k]=(img_float_I2[i][j][k]*log)%256
    return img_float_I2

#======================================================================
##PROBLEM:    CHECKING THE PRESENCE OF THE MI AND DETERMINANT
##STATUS:Runs Successfully
def check_matrix(test_matrix):
    
    determinant,i=check_determinant(test_matrix)
    determinant,MI,j=check_MI(determinant)
    determinant=determinant%256
    if i==1 and j==1:
        actual_matrix=test_matrix

    else:
        actual_matrix=np.array([
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                ])
        determinant= -1
        MI= -1
    return actual_matrix,MI,determinant
#======================================================================
##PROBLEM:      CHECKING THE PRESENCE OF THE POSITIVE DETERMINANT GREATER THAN ZERO
##STATUS:Runs Successfully
def check_determinant(key_actual):
    arr = np.array(key_actual)
    # calculate the determinant
    det = np.linalg.det(arr)
    
    if det==0:
        return 0,0
    else:
        return det,1

#======================================================================
##PROBLEM:      CHECKING THE PRESENCE OF THE MULTIPLICATIVE INVERSE
##STATUS:Runs Successfully
    
def check_MI(determinant):
    
    a = int(determinant)
    m = int(256)
    
    if a<-1:
        a=a%256
    inverse = find_inverse(a, m)
    if inverse is None:
        return a,0,0
    else:
        return a,inverse,1

def euclidean_algorithm(a, b):
    if b == 0:
        return a
    else:
        return euclidean_algorithm(b, a % b)

def find_inverse(a, m):
    gcd = euclidean_algorithm(a, m)
    if gcd != 1:
        return None  # The inverse does not exist
    else:
        for i in range(1, m):
            if (a * i) % m == 1:
                return i
#======================================================================
##PROBLEM:      KEY MATRIX GENERATION
##STATUS:Runs Successfully
def get_key(key):
    key_matrix =[
                    [key-10,key+8,key-15],
                    [key-3,key-0,key-6],
                    [key-20,key-17,key-1],
                ]
    key_matrix=np.array(key_matrix)
    key_matrix=key_matrix%256
    
    return key_matrix
#======================================================================
##PROBLEM:      TRANSPOSITION MATRIX GENERATION
##STATUS:Runs Successfully
def get_transpose(i):
    transpose_matrix=[
                        [i-10,i+8,i-15],
                        [i-3,i-0,i-6],
                        [i+4,i+1,i-1],
                    ]
    
    transpose_matrix=np.array(transpose_matrix)
    transpose_matrix=transpose_matrix%256

    return transpose_matrix
#======================================================================
def generate_random_prime():
    while True:
        num = random.randint(500, 1000)
        if sympy.isprime(num):
            return num
#======================================================================
##PROBLEM:      GETTING ALL REQUIREMENT FOR KEY PROCESSING
##STATUS:Runs Successfully
def method_RSA(key):
    # Generate a random prime number
    random_prime_1= generate_random_prime()
    random_prime_2 = generate_random_prime()
    
    p=min(random_prime_1,random_prime_2)
    q=max(random_prime_1,random_prime_2)

    n=p*q
    shy_fn=(p-1)*(q-1)
    possible_nos=[]
    new_possible_nos=[]
    
    for i in range(2,20):
        if ((i%p)!= 0)and((i%q)!= 0) :
            possible_nos.append(i)        
    
    for item in possible_nos:
        gcd = math.gcd(item,shy_fn)
        if gcd == 1:
            new_possible_nos.append(item)        
    return n,key,new_possible_nos,shy_fn
#======================================================================
def encrypt_key(n,key,public_key,shy_fn):
    if key>n:
        print(f"Key size is too large.\n{key} is larger than {n}.")
        return
    encrypt_key=(key**public_key)%n
    return encrypt_key
#======================================================================
def decrypt_key(n,key,public_key,encrypt_key,shy_fn):
    MI_Key,status=RSA_MI(public_key,shy_fn)
    print(f"MI is {MI_Key}.")
    if key>n:
        print(f"Key size is too large.\n{key} is larger than {n}.")
        return
    decrypt_key=(encrypt_key**MI_Key)%n
    return decrypt_key
#======================================================================
##PROBLEM:      TO FIND MI OF THE KEY
##STATUS:Runs Successfully
def RSA_MI(first,second):
    a = first
    m = second    
    if a<-1:
        a=a%m
    inverse = find_inverse(a, m)
    if inverse is None:
        print("MI not exist")
        return 0,-1
    else:
        return inverse,1

#======================================================================
##PROBLEM:    CHECKING THE PERCENTAGE OF ACCURACY USING SOURCE BACKUP
##STATUS:Runs Successfully
def check_percentage(height,width,img_float,back_up_float):
    correct=0
    incorrect=0
    
    for i in range(height):
        for j in range(width):
            for k in range(3):
                if img_float[i][j][k]==back_up_float[i][j][k]:
                    correct=correct+1
                else:
                    incorrect=incorrect+1

    percentage=(correct/(height*width*3))*100
    print(f"\nRecovery percentage is {percentage}.")

#======================================================================
##PROBLEM:    MIXING THE SOURCE WITH GIVEN MATRIX
##STATUS:Runs Successfully
def do_mixing(height,width,img_float,matrix):
    print(f"The input is {img_float[0][0]}.")
    for i in range(height):
        for j in range(width):
            A=np.array(img_float[i][j])
            B=np.array(matrix)
    
            C = np.dot(A, B)
            img_float[i][j]=C
            for k in range(3):
                img_float[i][j][k]=math.floor(img_float[i][j][k])
            img_float[i][j]=img_float[i][j]%256
    print(f"The ouput is {img_float[0][0]}.")
    return img_float
#======================================================================
##PROBLEM:      TO CALCULATE THE NPCR OF TWO IMAGES
##STATUS:Runs Successfully
def calculate_NPCR(image1_path, image2_path):
    # Load input images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Check if images have same dimensions
    if image1.size != image2.size:
        raise ValueError("Input images must have same dimensions")

    # Calculate NPCR value
    width, height = image1.size
    num_pixels = width * height
    diff_pixels = 0

    for x in range(width):
        for y in range(height):
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            if pixel1 != pixel2:
                diff_pixels += 1
    npcr = (diff_pixels / num_pixels) * 100
    return npcr
#======================================================================
##PROBLEM:      TO CALCULATE THE UACI OF TWO IMAGES
##STATUS:Runs Successfully
def calculate_UACI(image1_path, image2_path):
    # Load input images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Check if images have same dimensions
    if image1.size != image2.size:
        raise ValueError("Input images must have same dimensions")

    # Calculate NPCR value
    width, height = image1.size
    num_pixels = 255*width * height

    UACI_list=[0]
    for x in range(width):
        for y in range(height):
            pixel1 = list(image1.getpixel((x, y)))
            pixel2 = list(image2.getpixel((x, y)))
            total_sum1=pixel1[0]+pixel1[1]+pixel1[2]
            total_sum2=pixel2[0]+pixel2[1]+pixel2[2]

            UACI_list.append(abs(total_sum2-total_sum1))
    sum_UACI=(sum(UACI_list))

    UACI=(sum_UACI/num_pixels)*100
    
    return UACI
#======================================================================
##PROBLEM:    FIND THE INVERSE OF THE GIVEN MATRIX
##STATUS:Runs Successfully
def find_inverse_matrix(multi_no,matrix):

    adjoint=find_adjoint(matrix)
    matrix=multi_no*adjoint
    matrix=matrix%256
    
    return matrix
#======================================================================
##PROBLEM:    FIND THE ADJOINT
##STATUS:Runs Successfully
def find_adjoint(matrix):
    
    # Create a 3x3 matrix
    A=matrix
    # Calculate the matrix of minors
    minors = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            sub_matrix = A[np.array(list(range(i))+list(range(i+1,3)))[:,np.newaxis],
                           np.array(list(range(j))+list(range(j+1,3)))]
            minors[i, j] = (-1)**(i+j) * np.linalg.det(sub_matrix)
    
    # Transpose the matrix of minors to get the matrix of cofactors
    cofactors = minors.T
        
    # Calculate the adjoint of the matrix
    adj_A = cofactors
    
    return adj_A
#======================================================================
print("\t\t\t!!!!!!     BLIND SOURCE SEPARATION(IMAGE)      !!!!!! ")
print("Please Press Enter to Capture Image")
input()
capture_image()

image='Original.jpg'

img_float_I1=get_image(image)
img_float_I2=get_image(image)

back_up_float=get_image(image)

width, height=get_size(image)
print(f"The size of the image is {width}x{height} pixels.")

#Encryption
print("\n\t\t\t!!!!!!   ENCRYPTION PROCESS   !!!!!! ")

key=int(input("Please enter the key:"))

#Saving the plain images
set_image('Original(I1).jpg',img_float_I1)
print("I1 has been saved!!!!!\n")

img_float_I2=get_new_image(height,width,img_float_I2,key)

set_image('Original(I2).jpg',img_float_I2)
print("I2 has been saved!!!!!\n")

#Getting the key matrix
key_matrix=get_key(key)
key_final,MI_key,determinant_key=check_matrix(key_matrix)
temp=key
while(MI_key == -1):
    temp=temp-1
    key_matrix=get_key(temp)
    key_final,MI_key,determinant_key=check_matrix(key_matrix)
        
print(f"The key value is {temp}.")
print("FINAL KEY IS")        
print(key_final)
print()

#Getting the transpose matrix
MI_transpose= -1
while MI_transpose== -1:
    num = random.randint(500, 1000)
    transpose_matrix=get_transpose(num)
    transpose_final,MI_transpose,determinant_transpose=check_matrix(transpose_matrix)
print(f"Num Value is {num}.")
print("FINAL TRANSPOSE  IS")                
print(transpose_final)
print()

#Mixing process
#for C1
print("\nEncrypting the C1....")
print("Mixing the Source and Key......")
img_float_I1=do_mixing(height,width,img_float_I1,key_final)
print("Finished")
    
print("\nMixing the Source Key and Transpose Matrix......")
img_float_I1=do_mixing(height,width,img_float_I1,transpose_final)
print("Finished")

print("\nConverting back to Image(Cipher).......")
print("C1 has been saved!!!!!")
set_image('Encrypted(C1).jpg',img_float_I1)

#for C2
print("\nEncrypting the C2....")
print("Mixing the Source and Key......")
img_float_I2=do_mixing(height,width,img_float_I2,key_final)
print("Finished")
    
print("\nMixing the Source Key and Transpose Matrix......")
img_float_I2=do_mixing(height,width,img_float_I2,transpose_final)
print("Finished")

print("\nConverting back to Image(Cipher).......")
print("C2 has been saved!!!!!")
set_image('Encrypted(C2).jpg',img_float_I2)
print("\nEncryption process has been done successfully!!!!!!\n")

#Cryptanalysis
print("\nPress Enter to do Cryptanalysis......")
input()

NPCR=calculate_NPCR('encrypted(C1).jpg','encrypted(C2).jpg')
print("NPCR value:", NPCR)

UACI=calculate_UACI('encrypted(C1).jpg','encrypted(C2).jpg')
print("UACI value:",UACI)


#Getting the Public Key
n,key,new_possible_nos,shy_fn=method_RSA(key)

status= -1
while status== -1:
    public_key= random.choice(new_possible_nos)
    MI_Key,status=RSA_MI(public_key,shy_fn)
    new_possible_nos.remove(public_key)
    
print(f"\nPublic Key is {public_key}.")

encrypt_key=encrypt_key(n,key,public_key,shy_fn)
print(f"Encrypted key is {encrypt_key}.")


#Decryption
print("\n\t\t\t!!!!!!   DECRYPTION PROCESS   !!!!!! \n")
encrypt_key=int(input("Please enter the key:"))
decrypt_key=decrypt_key(n,key,public_key,encrypt_key,shy_fn)
print(f"Decrypted key is {decrypt_key}.")

#Getting the key matrix
key_matrix=get_key(decrypt_key)
key_final,MI_key,determinant_key=check_matrix(key_matrix)
temp=decrypt_key
while(MI_key == -1):
    temp=temp-1
    key_matrix=get_key(temp)
    key_final,MI_key,determinant_key=check_matrix(key_matrix)
        
print(f"The key value is {temp}.")
print("FINAL KEY IS")        
print(key_final)
print()

print("FINAL TRANSPOSE  IS")                
print(transpose_final)
print()

print("\nCalculating the Inverse..........")
key_inverse=find_inverse_matrix(MI_key,key_final)
print("\nKEY INVERSE")
print(key_inverse)

transpose_inverse=find_inverse_matrix(MI_transpose,transpose_final)
print("\nTRANSPOSE INVERSE")
print(transpose_inverse)


print("\nDemixing the Source Key and Transpose inverse Matrix......")
img_float_I1=do_mixing(height,width,img_float_I1,transpose_inverse)
print("Finished")    

print("\nDemixing the Source and Key inverse......")
img_float_I1=do_mixing(height,width,img_float_I1,key_inverse)
print("Finished")


print("\nConverting back to Image(Plain).......")
set_image('Decrypted.jpg',img_float_I1)
print("Plain Image has been saved!!!!!\n")


#Accuracy Checking
print("\n\t\t!!!!!!     ACCURACY CALCULATION      !!!!!! ")
print("Calculating the percentage of accuracy........")

if np.array_equal(img_float_I1,back_up_float):    
    print("\nHurrah!!!!\nImage has been recovered with 100 percent precision!!!!")
else:
    print("Hold for a sec........")
    print("\nOops!!!\nNot 100 percent!!!!!")
    check_percentage(height,width,img_float_I1,back_up_float)
    
print("\n\t\t!!!! END OF THE PROGRAM !!!!")

#======================================================================
