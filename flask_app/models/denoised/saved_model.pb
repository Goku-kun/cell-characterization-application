·
Ì£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-0-gb36436b0878î


conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
: *
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
: *
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
: *
dtype0

conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
: *
dtype0

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
: *
dtype0

conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
: *
dtype0

conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
: *
dtype0

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
: *
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:*
dtype0
n
Adadelta/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
_output_shapes
: *
dtype0	
p
Adadelta/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
_output_shapes
: *
dtype0

Adadelta/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
_output_shapes
: *
dtype0
l
Adadelta/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/rho
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
¬
$Adadelta/conv2d_14/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adadelta/conv2d_14/kernel/accum_grad
¥
8Adadelta/conv2d_14/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_14/kernel/accum_grad*&
_output_shapes
: *
dtype0

"Adadelta/conv2d_14/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adadelta/conv2d_14/bias/accum_grad

6Adadelta/conv2d_14/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_14/bias/accum_grad*
_output_shapes
: *
dtype0
¬
$Adadelta/conv2d_15/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$Adadelta/conv2d_15/kernel/accum_grad
¥
8Adadelta/conv2d_15/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_15/kernel/accum_grad*&
_output_shapes
:  *
dtype0

"Adadelta/conv2d_15/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adadelta/conv2d_15/bias/accum_grad

6Adadelta/conv2d_15/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_15/bias/accum_grad*
_output_shapes
: *
dtype0
¬
$Adadelta/conv2d_16/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$Adadelta/conv2d_16/kernel/accum_grad
¥
8Adadelta/conv2d_16/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_16/kernel/accum_grad*&
_output_shapes
:  *
dtype0

"Adadelta/conv2d_16/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adadelta/conv2d_16/bias/accum_grad

6Adadelta/conv2d_16/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_16/bias/accum_grad*
_output_shapes
: *
dtype0
¬
$Adadelta/conv2d_17/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$Adadelta/conv2d_17/kernel/accum_grad
¥
8Adadelta/conv2d_17/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_17/kernel/accum_grad*&
_output_shapes
:  *
dtype0

"Adadelta/conv2d_17/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adadelta/conv2d_17/bias/accum_grad

6Adadelta/conv2d_17/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_17/bias/accum_grad*
_output_shapes
: *
dtype0
¬
$Adadelta/conv2d_18/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$Adadelta/conv2d_18/kernel/accum_grad
¥
8Adadelta/conv2d_18/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_18/kernel/accum_grad*&
_output_shapes
:  *
dtype0

"Adadelta/conv2d_18/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adadelta/conv2d_18/bias/accum_grad

6Adadelta/conv2d_18/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_18/bias/accum_grad*
_output_shapes
: *
dtype0
¬
$Adadelta/conv2d_19/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$Adadelta/conv2d_19/kernel/accum_grad
¥
8Adadelta/conv2d_19/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_19/kernel/accum_grad*&
_output_shapes
:  *
dtype0

"Adadelta/conv2d_19/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adadelta/conv2d_19/bias/accum_grad

6Adadelta/conv2d_19/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_19/bias/accum_grad*
_output_shapes
: *
dtype0
¬
$Adadelta/conv2d_20/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adadelta/conv2d_20/kernel/accum_grad
¥
8Adadelta/conv2d_20/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_20/kernel/accum_grad*&
_output_shapes
: *
dtype0

"Adadelta/conv2d_20/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adadelta/conv2d_20/bias/accum_grad

6Adadelta/conv2d_20/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_20/bias/accum_grad*
_output_shapes
:*
dtype0
ª
#Adadelta/conv2d_14/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adadelta/conv2d_14/kernel/accum_var
£
7Adadelta/conv2d_14/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_14/kernel/accum_var*&
_output_shapes
: *
dtype0

!Adadelta/conv2d_14/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d_14/bias/accum_var

5Adadelta/conv2d_14/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_14/bias/accum_var*
_output_shapes
: *
dtype0
ª
#Adadelta/conv2d_15/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adadelta/conv2d_15/kernel/accum_var
£
7Adadelta/conv2d_15/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_15/kernel/accum_var*&
_output_shapes
:  *
dtype0

!Adadelta/conv2d_15/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d_15/bias/accum_var

5Adadelta/conv2d_15/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_15/bias/accum_var*
_output_shapes
: *
dtype0
ª
#Adadelta/conv2d_16/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adadelta/conv2d_16/kernel/accum_var
£
7Adadelta/conv2d_16/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_16/kernel/accum_var*&
_output_shapes
:  *
dtype0

!Adadelta/conv2d_16/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d_16/bias/accum_var

5Adadelta/conv2d_16/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_16/bias/accum_var*
_output_shapes
: *
dtype0
ª
#Adadelta/conv2d_17/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adadelta/conv2d_17/kernel/accum_var
£
7Adadelta/conv2d_17/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_17/kernel/accum_var*&
_output_shapes
:  *
dtype0

!Adadelta/conv2d_17/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d_17/bias/accum_var

5Adadelta/conv2d_17/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_17/bias/accum_var*
_output_shapes
: *
dtype0
ª
#Adadelta/conv2d_18/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adadelta/conv2d_18/kernel/accum_var
£
7Adadelta/conv2d_18/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_18/kernel/accum_var*&
_output_shapes
:  *
dtype0

!Adadelta/conv2d_18/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d_18/bias/accum_var

5Adadelta/conv2d_18/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_18/bias/accum_var*
_output_shapes
: *
dtype0
ª
#Adadelta/conv2d_19/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adadelta/conv2d_19/kernel/accum_var
£
7Adadelta/conv2d_19/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_19/kernel/accum_var*&
_output_shapes
:  *
dtype0

!Adadelta/conv2d_19/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d_19/bias/accum_var

5Adadelta/conv2d_19/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_19/bias/accum_var*
_output_shapes
: *
dtype0
ª
#Adadelta/conv2d_20/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adadelta/conv2d_20/kernel/accum_var
£
7Adadelta/conv2d_20/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_20/kernel/accum_var*&
_output_shapes
: *
dtype0

!Adadelta/conv2d_20/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adadelta/conv2d_20/bias/accum_var

5Adadelta/conv2d_20/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_20/bias/accum_var*
_output_shapes
:*
dtype0

NoOpNoOp
¸]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ó\
valueé\Bæ\ Bß\
Ó
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.trainable_variables
/regularization_losses
0	variables
1	keras_api
h

2kernel
3bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
R
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
R
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
R
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
h

Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
·
Viter
	Wdecay
Xlearning_rate
Yrho
accum_grad«
accum_grad¬
accum_grad­
accum_grad®(
accum_grad¯)
accum_grad°2
accum_grad±3
accum_grad²<
accum_grad³=
accum_grad´F
accum_gradµG
accum_grad¶P
accum_grad·Q
accum_grad¸	accum_var¹	accum_varº	accum_var»	accum_var¼(	accum_var½)	accum_var¾2	accum_var¿3	accum_varÀ<	accum_varÁ=	accum_varÂF	accum_varÃG	accum_varÄP	accum_varÅQ	accum_varÆ
f
0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13
 
f
0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13
­
Zlayer_metrics
[metrics
trainable_variables
regularization_losses
	variables
\layer_regularization_losses
]non_trainable_variables

^layers
 
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
_layer_metrics
`metrics
trainable_variables
regularization_losses
	variables
alayer_regularization_losses
bnon_trainable_variables

clayers
 
 
 
­
dlayer_metrics
emetrics
trainable_variables
regularization_losses
	variables
flayer_regularization_losses
gnon_trainable_variables

hlayers
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
ilayer_metrics
jmetrics
 trainable_variables
!regularization_losses
"	variables
klayer_regularization_losses
lnon_trainable_variables

mlayers
 
 
 
­
nlayer_metrics
ometrics
$trainable_variables
%regularization_losses
&	variables
player_regularization_losses
qnon_trainable_variables

rlayers
\Z
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
­
slayer_metrics
tmetrics
*trainable_variables
+regularization_losses
,	variables
ulayer_regularization_losses
vnon_trainable_variables

wlayers
 
 
 
­
xlayer_metrics
ymetrics
.trainable_variables
/regularization_losses
0	variables
zlayer_regularization_losses
{non_trainable_variables

|layers
\Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
 

20
31
¯
}layer_metrics
~metrics
4trainable_variables
5regularization_losses
6	variables
layer_regularization_losses
non_trainable_variables
layers
 
 
 
²
layer_metrics
metrics
8trainable_variables
9regularization_losses
:	variables
 layer_regularization_losses
non_trainable_variables
layers
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
²
layer_metrics
metrics
>trainable_variables
?regularization_losses
@	variables
 layer_regularization_losses
non_trainable_variables
layers
 
 
 
²
layer_metrics
metrics
Btrainable_variables
Cregularization_losses
D	variables
 layer_regularization_losses
non_trainable_variables
layers
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 

F0
G1
²
layer_metrics
metrics
Htrainable_variables
Iregularization_losses
J	variables
 layer_regularization_losses
non_trainable_variables
layers
 
 
 
²
layer_metrics
metrics
Ltrainable_variables
Mregularization_losses
N	variables
 layer_regularization_losses
non_trainable_variables
layers
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
 

P0
Q1
²
layer_metrics
metrics
Rtrainable_variables
Sregularization_losses
T	variables
 layer_regularization_losses
non_trainable_variables
layers
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 

 0
¡1
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

¢total

£count
¤	variables
¥	keras_api
I

¦total

§count
¨
_fn_kwargs
©	variables
ª	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

¢0
£1

¤	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

¦0
§1

©	variables

VARIABLE_VALUE$Adadelta/conv2d_14/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/conv2d_14/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/conv2d_15/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/conv2d_15/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/conv2d_16/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/conv2d_16/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/conv2d_17/kernel/accum_grad[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/conv2d_17/bias/accum_gradYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/conv2d_18/kernel/accum_grad[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/conv2d_18/bias/accum_gradYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/conv2d_19/kernel/accum_grad[layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/conv2d_19/bias/accum_gradYlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/conv2d_20/kernel/accum_grad[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/conv2d_20/bias/accum_gradYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/conv2d_14/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adadelta/conv2d_14/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/conv2d_15/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adadelta/conv2d_15/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/conv2d_16/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adadelta/conv2d_16/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/conv2d_17/kernel/accum_varZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adadelta/conv2d_17/bias/accum_varXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/conv2d_18/kernel/accum_varZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adadelta/conv2d_18/bias/accum_varXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/conv2d_19/kernel/accum_varZlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adadelta/conv2d_19/bias/accum_varXlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/conv2d_20/kernel/accum_varZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adadelta/conv2d_20/bias/accum_varXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_14_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ22
Î
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_14_inputconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_25638
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp8Adadelta/conv2d_14/kernel/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_14/bias/accum_grad/Read/ReadVariableOp8Adadelta/conv2d_15/kernel/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_15/bias/accum_grad/Read/ReadVariableOp8Adadelta/conv2d_16/kernel/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_16/bias/accum_grad/Read/ReadVariableOp8Adadelta/conv2d_17/kernel/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_17/bias/accum_grad/Read/ReadVariableOp8Adadelta/conv2d_18/kernel/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_18/bias/accum_grad/Read/ReadVariableOp8Adadelta/conv2d_19/kernel/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_19/bias/accum_grad/Read/ReadVariableOp8Adadelta/conv2d_20/kernel/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_20/bias/accum_grad/Read/ReadVariableOp7Adadelta/conv2d_14/kernel/accum_var/Read/ReadVariableOp5Adadelta/conv2d_14/bias/accum_var/Read/ReadVariableOp7Adadelta/conv2d_15/kernel/accum_var/Read/ReadVariableOp5Adadelta/conv2d_15/bias/accum_var/Read/ReadVariableOp7Adadelta/conv2d_16/kernel/accum_var/Read/ReadVariableOp5Adadelta/conv2d_16/bias/accum_var/Read/ReadVariableOp7Adadelta/conv2d_17/kernel/accum_var/Read/ReadVariableOp5Adadelta/conv2d_17/bias/accum_var/Read/ReadVariableOp7Adadelta/conv2d_18/kernel/accum_var/Read/ReadVariableOp5Adadelta/conv2d_18/bias/accum_var/Read/ReadVariableOp7Adadelta/conv2d_19/kernel/accum_var/Read/ReadVariableOp5Adadelta/conv2d_19/bias/accum_var/Read/ReadVariableOp7Adadelta/conv2d_20/kernel/accum_var/Read/ReadVariableOp5Adadelta/conv2d_20/bias/accum_var/Read/ReadVariableOpConst*?
Tin8
624	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_26174

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_1$Adadelta/conv2d_14/kernel/accum_grad"Adadelta/conv2d_14/bias/accum_grad$Adadelta/conv2d_15/kernel/accum_grad"Adadelta/conv2d_15/bias/accum_grad$Adadelta/conv2d_16/kernel/accum_grad"Adadelta/conv2d_16/bias/accum_grad$Adadelta/conv2d_17/kernel/accum_grad"Adadelta/conv2d_17/bias/accum_grad$Adadelta/conv2d_18/kernel/accum_grad"Adadelta/conv2d_18/bias/accum_grad$Adadelta/conv2d_19/kernel/accum_grad"Adadelta/conv2d_19/bias/accum_grad$Adadelta/conv2d_20/kernel/accum_grad"Adadelta/conv2d_20/bias/accum_grad#Adadelta/conv2d_14/kernel/accum_var!Adadelta/conv2d_14/bias/accum_var#Adadelta/conv2d_15/kernel/accum_var!Adadelta/conv2d_15/bias/accum_var#Adadelta/conv2d_16/kernel/accum_var!Adadelta/conv2d_16/bias/accum_var#Adadelta/conv2d_17/kernel/accum_var!Adadelta/conv2d_17/bias/accum_var#Adadelta/conv2d_18/kernel/accum_var!Adadelta/conv2d_18/bias/accum_var#Adadelta/conv2d_19/kernel/accum_var!Adadelta/conv2d_19/bias/accum_var#Adadelta/conv2d_20/kernel/accum_var!Adadelta/conv2d_20/bias/accum_var*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_26334Ä
£E

 __inference__wrapped_model_25130
conv2d_14_input9
5sequential_2_conv2d_14_conv2d_readvariableop_resource:
6sequential_2_conv2d_14_biasadd_readvariableop_resource9
5sequential_2_conv2d_15_conv2d_readvariableop_resource:
6sequential_2_conv2d_15_biasadd_readvariableop_resource9
5sequential_2_conv2d_16_conv2d_readvariableop_resource:
6sequential_2_conv2d_16_biasadd_readvariableop_resource9
5sequential_2_conv2d_17_conv2d_readvariableop_resource:
6sequential_2_conv2d_17_biasadd_readvariableop_resource9
5sequential_2_conv2d_18_conv2d_readvariableop_resource:
6sequential_2_conv2d_18_biasadd_readvariableop_resource9
5sequential_2_conv2d_19_conv2d_readvariableop_resource:
6sequential_2_conv2d_19_biasadd_readvariableop_resource9
5sequential_2_conv2d_20_conv2d_readvariableop_resource:
6sequential_2_conv2d_20_biasadd_readvariableop_resource
identityÚ
,sequential_2/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_2/conv2d_14/Conv2D/ReadVariableOpñ
sequential_2/conv2d_14/Conv2DConv2Dconv2d_14_input4sequential_2/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
sequential_2/conv2d_14/Conv2DÑ
-sequential_2/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_14/BiasAdd/ReadVariableOpä
sequential_2/conv2d_14/BiasAddBiasAdd&sequential_2/conv2d_14/Conv2D:output:05sequential_2/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2 
sequential_2/conv2d_14/BiasAdd­
sequential_2/activation_12/ReluRelu'sequential_2/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2!
sequential_2/activation_12/ReluÚ
,sequential_2/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_2/conv2d_15/Conv2D/ReadVariableOp
sequential_2/conv2d_15/Conv2DConv2D-sequential_2/activation_12/Relu:activations:04sequential_2/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
sequential_2/conv2d_15/Conv2DÑ
-sequential_2/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_15/BiasAdd/ReadVariableOpä
sequential_2/conv2d_15/BiasAddBiasAdd&sequential_2/conv2d_15/Conv2D:output:05sequential_2/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2 
sequential_2/conv2d_15/BiasAdd­
sequential_2/activation_13/ReluRelu'sequential_2/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2!
sequential_2/activation_13/ReluÚ
,sequential_2/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_2/conv2d_16/Conv2D/ReadVariableOp
sequential_2/conv2d_16/Conv2DConv2D-sequential_2/activation_13/Relu:activations:04sequential_2/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
sequential_2/conv2d_16/Conv2DÑ
-sequential_2/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_16/BiasAdd/ReadVariableOpä
sequential_2/conv2d_16/BiasAddBiasAdd&sequential_2/conv2d_16/Conv2D:output:05sequential_2/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2 
sequential_2/conv2d_16/BiasAdd­
sequential_2/activation_14/ReluRelu'sequential_2/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2!
sequential_2/activation_14/ReluÚ
,sequential_2/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_2/conv2d_17/Conv2D/ReadVariableOp
sequential_2/conv2d_17/Conv2DConv2D-sequential_2/activation_14/Relu:activations:04sequential_2/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
sequential_2/conv2d_17/Conv2DÑ
-sequential_2/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_17/BiasAdd/ReadVariableOpä
sequential_2/conv2d_17/BiasAddBiasAdd&sequential_2/conv2d_17/Conv2D:output:05sequential_2/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2 
sequential_2/conv2d_17/BiasAdd­
sequential_2/activation_15/ReluRelu'sequential_2/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2!
sequential_2/activation_15/ReluÚ
,sequential_2/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_2/conv2d_18/Conv2D/ReadVariableOp
sequential_2/conv2d_18/Conv2DConv2D-sequential_2/activation_15/Relu:activations:04sequential_2/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
sequential_2/conv2d_18/Conv2DÑ
-sequential_2/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_18/BiasAdd/ReadVariableOpä
sequential_2/conv2d_18/BiasAddBiasAdd&sequential_2/conv2d_18/Conv2D:output:05sequential_2/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2 
sequential_2/conv2d_18/BiasAdd­
sequential_2/activation_16/ReluRelu'sequential_2/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2!
sequential_2/activation_16/ReluÚ
,sequential_2/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_2/conv2d_19/Conv2D/ReadVariableOp
sequential_2/conv2d_19/Conv2DConv2D-sequential_2/activation_16/Relu:activations:04sequential_2/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
sequential_2/conv2d_19/Conv2DÑ
-sequential_2/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_19/BiasAdd/ReadVariableOpä
sequential_2/conv2d_19/BiasAddBiasAdd&sequential_2/conv2d_19/Conv2D:output:05sequential_2/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2 
sequential_2/conv2d_19/BiasAdd­
sequential_2/activation_17/ReluRelu'sequential_2/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2!
sequential_2/activation_17/ReluÚ
,sequential_2/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_2/conv2d_20/Conv2D/ReadVariableOp
sequential_2/conv2d_20/Conv2DConv2D-sequential_2/activation_17/Relu:activations:04sequential_2/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
sequential_2/conv2d_20/Conv2DÑ
-sequential_2/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_2/conv2d_20/BiasAdd/ReadVariableOpä
sequential_2/conv2d_20/BiasAddBiasAdd&sequential_2/conv2d_20/Conv2D:output:05sequential_2/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222 
sequential_2/conv2d_20/BiasAdd
IdentityIdentity'sequential_2/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22:::::::::::::::` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
)
_user_specified_nameconv2d_14_input
Ö
d
H__inference_activation_14_layer_call_and_return_conditional_losses_25890

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_17_layer_call_and_return_conditional_losses_25261

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_15_layer_call_and_return_conditional_losses_25282

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_16_layer_call_and_return_conditional_losses_25948

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
·9

G__inference_sequential_2_layer_call_and_return_conditional_losses_25440
conv2d_14_input
conv2d_14_25398
conv2d_14_25400
conv2d_15_25404
conv2d_15_25406
conv2d_16_25410
conv2d_16_25412
conv2d_17_25416
conv2d_17_25418
conv2d_18_25422
conv2d_18_25424
conv2d_19_25428
conv2d_19_25430
conv2d_20_25434
conv2d_20_25436
identity¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCallª
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputconv2d_14_25398conv2d_14_25400*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_251442#
!conv2d_14/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_251652
activation_12/PartitionedCallÁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_15_25404conv2d_15_25406*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_251832#
!conv2d_15/StatefulPartitionedCall
activation_13/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_252042
activation_13/PartitionedCallÁ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_13/PartitionedCall:output:0conv2d_16_25410conv2d_16_25412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_252222#
!conv2d_16/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_252432
activation_14/PartitionedCallÁ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0conv2d_17_25416conv2d_17_25418*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_252612#
!conv2d_17/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_252822
activation_15/PartitionedCallÁ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0conv2d_18_25422conv2d_18_25424*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_253002#
!conv2d_18/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_253212
activation_16/PartitionedCallÁ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_19_25428conv2d_19_25430*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_253392#
!conv2d_19/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_253602
activation_17/PartitionedCallÁ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0conv2d_20_25434conv2d_20_25436*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_253782#
!conv2d_20/StatefulPartitionedCall
IdentityIdentity*conv2d_20/StatefulPartitionedCall:output:0"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
)
_user_specified_nameconv2d_14_input


¸
,__inference_sequential_2_layer_call_fn_25775

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_254882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
§

Á
,__inference_sequential_2_layer_call_fn_25519
conv2d_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_254882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
)
_user_specified_nameconv2d_14_input
ÿ
~
)__inference_conv2d_17_layer_call_fn_25914

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_252612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_15_layer_call_and_return_conditional_losses_25919

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
½
I
-__inference_activation_15_layer_call_fn_25924

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_252822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_20_layer_call_and_return_conditional_losses_25992

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¿Ú
ý
!__inference__traced_restore_26334
file_prefix%
!assignvariableop_conv2d_14_kernel%
!assignvariableop_1_conv2d_14_bias'
#assignvariableop_2_conv2d_15_kernel%
!assignvariableop_3_conv2d_15_bias'
#assignvariableop_4_conv2d_16_kernel%
!assignvariableop_5_conv2d_16_bias'
#assignvariableop_6_conv2d_17_kernel%
!assignvariableop_7_conv2d_17_bias'
#assignvariableop_8_conv2d_18_kernel%
!assignvariableop_9_conv2d_18_bias(
$assignvariableop_10_conv2d_19_kernel&
"assignvariableop_11_conv2d_19_bias(
$assignvariableop_12_conv2d_20_kernel&
"assignvariableop_13_conv2d_20_bias%
!assignvariableop_14_adadelta_iter&
"assignvariableop_15_adadelta_decay.
*assignvariableop_16_adadelta_learning_rate$
 assignvariableop_17_adadelta_rho
assignvariableop_18_total
assignvariableop_19_count
assignvariableop_20_total_1
assignvariableop_21_count_1<
8assignvariableop_22_adadelta_conv2d_14_kernel_accum_grad:
6assignvariableop_23_adadelta_conv2d_14_bias_accum_grad<
8assignvariableop_24_adadelta_conv2d_15_kernel_accum_grad:
6assignvariableop_25_adadelta_conv2d_15_bias_accum_grad<
8assignvariableop_26_adadelta_conv2d_16_kernel_accum_grad:
6assignvariableop_27_adadelta_conv2d_16_bias_accum_grad<
8assignvariableop_28_adadelta_conv2d_17_kernel_accum_grad:
6assignvariableop_29_adadelta_conv2d_17_bias_accum_grad<
8assignvariableop_30_adadelta_conv2d_18_kernel_accum_grad:
6assignvariableop_31_adadelta_conv2d_18_bias_accum_grad<
8assignvariableop_32_adadelta_conv2d_19_kernel_accum_grad:
6assignvariableop_33_adadelta_conv2d_19_bias_accum_grad<
8assignvariableop_34_adadelta_conv2d_20_kernel_accum_grad:
6assignvariableop_35_adadelta_conv2d_20_bias_accum_grad;
7assignvariableop_36_adadelta_conv2d_14_kernel_accum_var9
5assignvariableop_37_adadelta_conv2d_14_bias_accum_var;
7assignvariableop_38_adadelta_conv2d_15_kernel_accum_var9
5assignvariableop_39_adadelta_conv2d_15_bias_accum_var;
7assignvariableop_40_adadelta_conv2d_16_kernel_accum_var9
5assignvariableop_41_adadelta_conv2d_16_bias_accum_var;
7assignvariableop_42_adadelta_conv2d_17_kernel_accum_var9
5assignvariableop_43_adadelta_conv2d_17_bias_accum_var;
7assignvariableop_44_adadelta_conv2d_18_kernel_accum_var9
5assignvariableop_45_adadelta_conv2d_18_bias_accum_var;
7assignvariableop_46_adadelta_conv2d_19_kernel_accum_var9
5assignvariableop_47_adadelta_conv2d_19_bias_accum_var;
7assignvariableop_48_adadelta_conv2d_20_kernel_accum_var9
5assignvariableop_49_adadelta_conv2d_20_bias_accum_var
identity_51¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¸
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*Ä
valueºB·3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesô
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*â
_output_shapesÏ
Ì:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_15_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_15_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_16_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_16_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_17_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_17_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_18_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_18_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_19_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_19_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¬
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_20_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_20_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_adadelta_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_adadelta_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16²
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adadelta_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¨
AssignVariableOp_17AssignVariableOp assignvariableop_17_adadelta_rhoIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22À
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adadelta_conv2d_14_kernel_accum_gradIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¾
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adadelta_conv2d_14_bias_accum_gradIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24À
AssignVariableOp_24AssignVariableOp8assignvariableop_24_adadelta_conv2d_15_kernel_accum_gradIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¾
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adadelta_conv2d_15_bias_accum_gradIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26À
AssignVariableOp_26AssignVariableOp8assignvariableop_26_adadelta_conv2d_16_kernel_accum_gradIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¾
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adadelta_conv2d_16_bias_accum_gradIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28À
AssignVariableOp_28AssignVariableOp8assignvariableop_28_adadelta_conv2d_17_kernel_accum_gradIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¾
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adadelta_conv2d_17_bias_accum_gradIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30À
AssignVariableOp_30AssignVariableOp8assignvariableop_30_adadelta_conv2d_18_kernel_accum_gradIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¾
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adadelta_conv2d_18_bias_accum_gradIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32À
AssignVariableOp_32AssignVariableOp8assignvariableop_32_adadelta_conv2d_19_kernel_accum_gradIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¾
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adadelta_conv2d_19_bias_accum_gradIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34À
AssignVariableOp_34AssignVariableOp8assignvariableop_34_adadelta_conv2d_20_kernel_accum_gradIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¾
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adadelta_conv2d_20_bias_accum_gradIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¿
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adadelta_conv2d_14_kernel_accum_varIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37½
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adadelta_conv2d_14_bias_accum_varIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¿
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adadelta_conv2d_15_kernel_accum_varIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39½
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adadelta_conv2d_15_bias_accum_varIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¿
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adadelta_conv2d_16_kernel_accum_varIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41½
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adadelta_conv2d_16_bias_accum_varIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¿
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adadelta_conv2d_17_kernel_accum_varIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43½
AssignVariableOp_43AssignVariableOp5assignvariableop_43_adadelta_conv2d_17_bias_accum_varIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¿
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adadelta_conv2d_18_kernel_accum_varIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45½
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adadelta_conv2d_18_bias_accum_varIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¿
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adadelta_conv2d_19_kernel_accum_varIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47½
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adadelta_conv2d_19_bias_accum_varIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¿
AssignVariableOp_48AssignVariableOp7assignvariableop_48_adadelta_conv2d_20_kernel_accum_varIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49½
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adadelta_conv2d_20_bias_accum_varIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_499
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpª	
Identity_50Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_50	
Identity_51IdentityIdentity_50:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_51"#
identity_51Identity_51:output:0*ß
_input_shapesÍ
Ê: ::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
½
I
-__inference_activation_16_layer_call_fn_25953

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_253212
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_15_layer_call_and_return_conditional_losses_25847

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_18_layer_call_fn_25943

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_253002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¢p
Ê
__inference__traced_save_26174
file_prefix/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopC
?savev2_adadelta_conv2d_14_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_14_bias_accum_grad_read_readvariableopC
?savev2_adadelta_conv2d_15_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_15_bias_accum_grad_read_readvariableopC
?savev2_adadelta_conv2d_16_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_16_bias_accum_grad_read_readvariableopC
?savev2_adadelta_conv2d_17_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_17_bias_accum_grad_read_readvariableopC
?savev2_adadelta_conv2d_18_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_18_bias_accum_grad_read_readvariableopC
?savev2_adadelta_conv2d_19_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_19_bias_accum_grad_read_readvariableopC
?savev2_adadelta_conv2d_20_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_20_bias_accum_grad_read_readvariableopB
>savev2_adadelta_conv2d_14_kernel_accum_var_read_readvariableop@
<savev2_adadelta_conv2d_14_bias_accum_var_read_readvariableopB
>savev2_adadelta_conv2d_15_kernel_accum_var_read_readvariableop@
<savev2_adadelta_conv2d_15_bias_accum_var_read_readvariableopB
>savev2_adadelta_conv2d_16_kernel_accum_var_read_readvariableop@
<savev2_adadelta_conv2d_16_bias_accum_var_read_readvariableopB
>savev2_adadelta_conv2d_17_kernel_accum_var_read_readvariableop@
<savev2_adadelta_conv2d_17_bias_accum_var_read_readvariableopB
>savev2_adadelta_conv2d_18_kernel_accum_var_read_readvariableop@
<savev2_adadelta_conv2d_18_bias_accum_var_read_readvariableopB
>savev2_adadelta_conv2d_19_kernel_accum_var_read_readvariableop@
<savev2_adadelta_conv2d_19_bias_accum_var_read_readvariableopB
>savev2_adadelta_conv2d_20_kernel_accum_var_read_readvariableop@
<savev2_adadelta_conv2d_20_bias_accum_var_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ddd21c18d951454fad41938e1dda6e7a/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename²
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*Ä
valueºB·3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesî
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop?savev2_adadelta_conv2d_14_kernel_accum_grad_read_readvariableop=savev2_adadelta_conv2d_14_bias_accum_grad_read_readvariableop?savev2_adadelta_conv2d_15_kernel_accum_grad_read_readvariableop=savev2_adadelta_conv2d_15_bias_accum_grad_read_readvariableop?savev2_adadelta_conv2d_16_kernel_accum_grad_read_readvariableop=savev2_adadelta_conv2d_16_bias_accum_grad_read_readvariableop?savev2_adadelta_conv2d_17_kernel_accum_grad_read_readvariableop=savev2_adadelta_conv2d_17_bias_accum_grad_read_readvariableop?savev2_adadelta_conv2d_18_kernel_accum_grad_read_readvariableop=savev2_adadelta_conv2d_18_bias_accum_grad_read_readvariableop?savev2_adadelta_conv2d_19_kernel_accum_grad_read_readvariableop=savev2_adadelta_conv2d_19_bias_accum_grad_read_readvariableop?savev2_adadelta_conv2d_20_kernel_accum_grad_read_readvariableop=savev2_adadelta_conv2d_20_bias_accum_grad_read_readvariableop>savev2_adadelta_conv2d_14_kernel_accum_var_read_readvariableop<savev2_adadelta_conv2d_14_bias_accum_var_read_readvariableop>savev2_adadelta_conv2d_15_kernel_accum_var_read_readvariableop<savev2_adadelta_conv2d_15_bias_accum_var_read_readvariableop>savev2_adadelta_conv2d_16_kernel_accum_var_read_readvariableop<savev2_adadelta_conv2d_16_bias_accum_var_read_readvariableop>savev2_adadelta_conv2d_17_kernel_accum_var_read_readvariableop<savev2_adadelta_conv2d_17_bias_accum_var_read_readvariableop>savev2_adadelta_conv2d_18_kernel_accum_var_read_readvariableop<savev2_adadelta_conv2d_18_bias_accum_var_read_readvariableop>savev2_adadelta_conv2d_19_kernel_accum_var_read_readvariableop<savev2_adadelta_conv2d_19_bias_accum_var_read_readvariableop>savev2_adadelta_conv2d_20_kernel_accum_var_read_readvariableop<savev2_adadelta_conv2d_20_bias_accum_var_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *A
dtypes7
523	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*¡
_input_shapes
: : : :  : :  : :  : :  : :  : : :: : : : : : : : : : :  : :  : :  : :  : :  : : :: : :  : :  : :  : :  : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
:  : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  :  

_output_shapes
: :,!(
&
_output_shapes
:  : "

_output_shapes
: :,#(
&
_output_shapes
: : $

_output_shapes
::,%(
&
_output_shapes
: : &

_output_shapes
: :,'(
&
_output_shapes
:  : (

_output_shapes
: :,)(
&
_output_shapes
:  : *

_output_shapes
: :,+(
&
_output_shapes
:  : ,

_output_shapes
: :,-(
&
_output_shapes
:  : .

_output_shapes
: :,/(
&
_output_shapes
:  : 0

_output_shapes
: :,1(
&
_output_shapes
: : 2

_output_shapes
::3

_output_shapes
: 
¥
¬
D__inference_conv2d_16_layer_call_and_return_conditional_losses_25876

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_14_layer_call_and_return_conditional_losses_25144

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
½
I
-__inference_activation_13_layer_call_fn_25866

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_252042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
9

G__inference_sequential_2_layer_call_and_return_conditional_losses_25488

inputs
conv2d_14_25446
conv2d_14_25448
conv2d_15_25452
conv2d_15_25454
conv2d_16_25458
conv2d_16_25460
conv2d_17_25464
conv2d_17_25466
conv2d_18_25470
conv2d_18_25472
conv2d_19_25476
conv2d_19_25478
conv2d_20_25482
conv2d_20_25484
identity¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¡
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_25446conv2d_14_25448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_251442#
!conv2d_14/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_251652
activation_12/PartitionedCallÁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_15_25452conv2d_15_25454*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_251832#
!conv2d_15/StatefulPartitionedCall
activation_13/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_252042
activation_13/PartitionedCallÁ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_13/PartitionedCall:output:0conv2d_16_25458conv2d_16_25460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_252222#
!conv2d_16/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_252432
activation_14/PartitionedCallÁ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0conv2d_17_25464conv2d_17_25466*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_252612#
!conv2d_17/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_252822
activation_15/PartitionedCallÁ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0conv2d_18_25470conv2d_18_25472*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_253002#
!conv2d_18/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_253212
activation_16/PartitionedCallÁ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_19_25476conv2d_19_25478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_253392#
!conv2d_19/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_253602
activation_17/PartitionedCallÁ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0conv2d_20_25482conv2d_20_25484*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_253782#
!conv2d_20/StatefulPartitionedCall
IdentityIdentity*conv2d_20/StatefulPartitionedCall:output:0"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_16_layer_call_and_return_conditional_losses_25222

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
§

Á
,__inference_sequential_2_layer_call_fn_25597
conv2d_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_255662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
)
_user_specified_nameconv2d_14_input
8
ñ
G__inference_sequential_2_layer_call_and_return_conditional_losses_25742

inputs,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource
identity³
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_14/Conv2D/ReadVariableOpÁ
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_14/Conv2Dª
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp°
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_14/BiasAdd
activation_12/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_12/Relu³
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_15/Conv2D/ReadVariableOpÛ
conv2d_15/Conv2DConv2D activation_12/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_15/Conv2Dª
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp°
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_15/BiasAdd
activation_13/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_13/Relu³
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_16/Conv2D/ReadVariableOpÛ
conv2d_16/Conv2DConv2D activation_13/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_16/Conv2Dª
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp°
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_16/BiasAdd
activation_14/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_14/Relu³
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_17/Conv2D/ReadVariableOpÛ
conv2d_17/Conv2DConv2D activation_14/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_17/Conv2Dª
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp°
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_17/BiasAdd
activation_15/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_15/Relu³
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_18/Conv2D/ReadVariableOpÛ
conv2d_18/Conv2DConv2D activation_15/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_18/Conv2Dª
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp°
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_18/BiasAdd
activation_16/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_16/Relu³
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_19/Conv2D/ReadVariableOpÛ
conv2d_19/Conv2DConv2D activation_16/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_19/Conv2Dª
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp°
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_19/BiasAdd
activation_17/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_17/Relu³
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_20/Conv2D/ReadVariableOpÛ
conv2d_20/Conv2DConv2D activation_17/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
conv2d_20/Conv2Dª
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp°
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
conv2d_20/BiasAddv
IdentityIdentityconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22:::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_14_layer_call_fn_25827

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_251442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_20_layer_call_and_return_conditional_losses_25378

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_17_layer_call_and_return_conditional_losses_25905

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
½
I
-__inference_activation_14_layer_call_fn_25895

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_252432
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_18_layer_call_and_return_conditional_losses_25300

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_15_layer_call_and_return_conditional_losses_25183

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_12_layer_call_and_return_conditional_losses_25832

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_20_layer_call_fn_26001

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_253782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_13_layer_call_and_return_conditional_losses_25204

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_19_layer_call_fn_25972

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_253392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_14_layer_call_and_return_conditional_losses_25818

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_15_layer_call_fn_25856

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_251832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_12_layer_call_and_return_conditional_losses_25165

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
·9

G__inference_sequential_2_layer_call_and_return_conditional_losses_25395
conv2d_14_input
conv2d_14_25155
conv2d_14_25157
conv2d_15_25194
conv2d_15_25196
conv2d_16_25233
conv2d_16_25235
conv2d_17_25272
conv2d_17_25274
conv2d_18_25311
conv2d_18_25313
conv2d_19_25350
conv2d_19_25352
conv2d_20_25389
conv2d_20_25391
identity¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCallª
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputconv2d_14_25155conv2d_14_25157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_251442#
!conv2d_14/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_251652
activation_12/PartitionedCallÁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_15_25194conv2d_15_25196*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_251832#
!conv2d_15/StatefulPartitionedCall
activation_13/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_252042
activation_13/PartitionedCallÁ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_13/PartitionedCall:output:0conv2d_16_25233conv2d_16_25235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_252222#
!conv2d_16/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_252432
activation_14/PartitionedCallÁ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0conv2d_17_25272conv2d_17_25274*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_252612#
!conv2d_17/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_252822
activation_15/PartitionedCallÁ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0conv2d_18_25311conv2d_18_25313*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_253002#
!conv2d_18/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_253212
activation_16/PartitionedCallÁ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_19_25350conv2d_19_25352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_253392#
!conv2d_19/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_253602
activation_17/PartitionedCallÁ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0conv2d_20_25389conv2d_20_25391*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_253782#
!conv2d_20/StatefulPartitionedCall
IdentityIdentity*conv2d_20/StatefulPartitionedCall:output:0"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
)
_user_specified_nameconv2d_14_input
Ö
d
H__inference_activation_17_layer_call_and_return_conditional_losses_25360

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_19_layer_call_and_return_conditional_losses_25339

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
8
ñ
G__inference_sequential_2_layer_call_and_return_conditional_losses_25690

inputs,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource
identity³
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_14/Conv2D/ReadVariableOpÁ
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_14/Conv2Dª
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp°
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_14/BiasAdd
activation_12/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_12/Relu³
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_15/Conv2D/ReadVariableOpÛ
conv2d_15/Conv2DConv2D activation_12/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_15/Conv2Dª
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp°
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_15/BiasAdd
activation_13/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_13/Relu³
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_16/Conv2D/ReadVariableOpÛ
conv2d_16/Conv2DConv2D activation_13/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_16/Conv2Dª
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp°
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_16/BiasAdd
activation_14/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_14/Relu³
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_17/Conv2D/ReadVariableOpÛ
conv2d_17/Conv2DConv2D activation_14/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_17/Conv2Dª
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp°
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_17/BiasAdd
activation_15/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_15/Relu³
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_18/Conv2D/ReadVariableOpÛ
conv2d_18/Conv2DConv2D activation_15/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_18/Conv2Dª
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp°
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_18/BiasAdd
activation_16/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_16/Relu³
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_19/Conv2D/ReadVariableOpÛ
conv2d_19/Conv2DConv2D activation_16/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
conv2d_19/Conv2Dª
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp°
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
conv2d_19/BiasAdd
activation_17/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
activation_17/Relu³
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_20/Conv2D/ReadVariableOpÛ
conv2d_20/Conv2DConv2D activation_17/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
2
conv2d_20/Conv2Dª
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp°
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
conv2d_20/BiasAddv
IdentityIdentityconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22:::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
÷	
¸
#__inference_signature_wrapper_25638
conv2d_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_251302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
)
_user_specified_nameconv2d_14_input
Ö
d
H__inference_activation_17_layer_call_and_return_conditional_losses_25977

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
½
I
-__inference_activation_12_layer_call_fn_25837

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_251652
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_16_layer_call_and_return_conditional_losses_25321

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
½
I
-__inference_activation_17_layer_call_fn_25982

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_253602
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_14_layer_call_and_return_conditional_losses_25243

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
Ö
d
H__inference_activation_13_layer_call_and_return_conditional_losses_25861

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_18_layer_call_and_return_conditional_losses_25934

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
9

G__inference_sequential_2_layer_call_and_return_conditional_losses_25566

inputs
conv2d_14_25524
conv2d_14_25526
conv2d_15_25530
conv2d_15_25532
conv2d_16_25536
conv2d_16_25538
conv2d_17_25542
conv2d_17_25544
conv2d_18_25548
conv2d_18_25550
conv2d_19_25554
conv2d_19_25556
conv2d_20_25560
conv2d_20_25562
identity¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall¢!conv2d_19/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¡
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_25524conv2d_14_25526*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_251442#
!conv2d_14/StatefulPartitionedCall
activation_12/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_251652
activation_12/PartitionedCallÁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_12/PartitionedCall:output:0conv2d_15_25530conv2d_15_25532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_251832#
!conv2d_15/StatefulPartitionedCall
activation_13/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_252042
activation_13/PartitionedCallÁ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_13/PartitionedCall:output:0conv2d_16_25536conv2d_16_25538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_252222#
!conv2d_16/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_252432
activation_14/PartitionedCallÁ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0conv2d_17_25542conv2d_17_25544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_252612#
!conv2d_17/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_252822
activation_15/PartitionedCallÁ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0conv2d_18_25548conv2d_18_25550*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_253002#
!conv2d_18/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_253212
activation_16/PartitionedCallÁ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_19_25554conv2d_19_25556*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_253392#
!conv2d_19/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_253602
activation_17/PartitionedCallÁ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0conv2d_20_25560conv2d_20_25562*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_253782#
!conv2d_20/StatefulPartitionedCall
IdentityIdentity*conv2d_20/StatefulPartitionedCall:output:0"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs


¸
,__inference_sequential_2_layer_call_fn_25808

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_255662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ22::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_16_layer_call_fn_25885

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_252222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs
¥
¬
D__inference_conv2d_19_layer_call_and_return_conditional_losses_25963

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ì
serving_default¸
S
conv2d_14_input@
!serving_default_conv2d_14_input:0ÿÿÿÿÿÿÿÿÿ22E
	conv2d_208
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ22tensorflow/serving/predict:ô
¶g
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Ç_default_save_signature
È__call__
+É&call_and_return_all_conditional_losses"c
_tf_keras_sequentialçb{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_14_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_14_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
÷


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"Ð	
_tf_keras_layer¶	{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 1]}}
Ù
trainable_variables
regularization_losses
	variables
	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"class_name": "Activation", "name": "activation_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}}
ø	

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ù
$trainable_variables
%regularization_losses
&	variables
'	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"class_name": "Activation", "name": "activation_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}}
ø	

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ù
.trainable_variables
/regularization_losses
0	variables
1	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"class_name": "Activation", "name": "activation_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}}
ù


2kernel
3bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"Ò	
_tf_keras_layer¸	{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ù
8trainable_variables
9regularization_losses
:	variables
;	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"class_name": "Activation", "name": "activation_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}}
ø	

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ù
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"class_name": "Activation", "name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}
ø	

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ù
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
à__call__
+á&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"class_name": "Activation", "name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}
÷	

Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ê
Viter
	Wdecay
Xlearning_rate
Yrho
accum_grad«
accum_grad¬
accum_grad­
accum_grad®(
accum_grad¯)
accum_grad°2
accum_grad±3
accum_grad²<
accum_grad³=
accum_grad´F
accum_gradµG
accum_grad¶P
accum_grad·Q
accum_grad¸	accum_var¹	accum_varº	accum_var»	accum_var¼(	accum_var½)	accum_var¾2	accum_var¿3	accum_varÀ<	accum_varÁ=	accum_varÂF	accum_varÃG	accum_varÄP	accum_varÅQ	accum_varÆ"
	optimizer

0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13"
trackable_list_wrapper
Î
Zlayer_metrics
[metrics
trainable_variables
regularization_losses
	variables
\layer_regularization_losses
]non_trainable_variables

^layers
È__call__
Ç_default_save_signature
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
-
äserving_default"
signature_map
*:( 2conv2d_14/kernel
: 2conv2d_14/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
_layer_metrics
`metrics
trainable_variables
regularization_losses
	variables
alayer_regularization_losses
bnon_trainable_variables

clayers
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
dlayer_metrics
emetrics
trainable_variables
regularization_losses
	variables
flayer_regularization_losses
gnon_trainable_variables

hlayers
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_15/kernel
: 2conv2d_15/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
ilayer_metrics
jmetrics
 trainable_variables
!regularization_losses
"	variables
klayer_regularization_losses
lnon_trainable_variables

mlayers
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
nlayer_metrics
ometrics
$trainable_variables
%regularization_losses
&	variables
player_regularization_losses
qnon_trainable_variables

rlayers
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_16/kernel
: 2conv2d_16/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
°
slayer_metrics
tmetrics
*trainable_variables
+regularization_losses
,	variables
ulayer_regularization_losses
vnon_trainable_variables

wlayers
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
xlayer_metrics
ymetrics
.trainable_variables
/regularization_losses
0	variables
zlayer_regularization_losses
{non_trainable_variables

|layers
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_17/kernel
: 2conv2d_17/bias
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
²
}layer_metrics
~metrics
4trainable_variables
5regularization_losses
6	variables
layer_regularization_losses
non_trainable_variables
layers
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
8trainable_variables
9regularization_losses
:	variables
 layer_regularization_losses
non_trainable_variables
layers
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_18/kernel
: 2conv2d_18/bias
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
µ
layer_metrics
metrics
>trainable_variables
?regularization_losses
@	variables
 layer_regularization_losses
non_trainable_variables
layers
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
Btrainable_variables
Cregularization_losses
D	variables
 layer_regularization_losses
non_trainable_variables
layers
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_19/kernel
: 2conv2d_19/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
µ
layer_metrics
metrics
Htrainable_variables
Iregularization_losses
J	variables
 layer_regularization_losses
non_trainable_variables
layers
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
Ltrainable_variables
Mregularization_losses
N	variables
 layer_regularization_losses
non_trainable_variables
layers
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_20/kernel
:2conv2d_20/bias
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
µ
layer_metrics
metrics
Rtrainable_variables
Sregularization_losses
T	variables
 layer_regularization_losses
non_trainable_variables
layers
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
 "
trackable_dict_wrapper
0
 0
¡1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

¢total

£count
¤	variables
¥	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ

¦total

§count
¨
_fn_kwargs
©	variables
ª	keras_api"³
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
¢0
£1"
trackable_list_wrapper
.
¤	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¦0
§1"
trackable_list_wrapper
.
©	variables"
_generic_user_object
<:: 2$Adadelta/conv2d_14/kernel/accum_grad
.:, 2"Adadelta/conv2d_14/bias/accum_grad
<::  2$Adadelta/conv2d_15/kernel/accum_grad
.:, 2"Adadelta/conv2d_15/bias/accum_grad
<::  2$Adadelta/conv2d_16/kernel/accum_grad
.:, 2"Adadelta/conv2d_16/bias/accum_grad
<::  2$Adadelta/conv2d_17/kernel/accum_grad
.:, 2"Adadelta/conv2d_17/bias/accum_grad
<::  2$Adadelta/conv2d_18/kernel/accum_grad
.:, 2"Adadelta/conv2d_18/bias/accum_grad
<::  2$Adadelta/conv2d_19/kernel/accum_grad
.:, 2"Adadelta/conv2d_19/bias/accum_grad
<:: 2$Adadelta/conv2d_20/kernel/accum_grad
.:,2"Adadelta/conv2d_20/bias/accum_grad
;:9 2#Adadelta/conv2d_14/kernel/accum_var
-:+ 2!Adadelta/conv2d_14/bias/accum_var
;:9  2#Adadelta/conv2d_15/kernel/accum_var
-:+ 2!Adadelta/conv2d_15/bias/accum_var
;:9  2#Adadelta/conv2d_16/kernel/accum_var
-:+ 2!Adadelta/conv2d_16/bias/accum_var
;:9  2#Adadelta/conv2d_17/kernel/accum_var
-:+ 2!Adadelta/conv2d_17/bias/accum_var
;:9  2#Adadelta/conv2d_18/kernel/accum_var
-:+ 2!Adadelta/conv2d_18/bias/accum_var
;:9  2#Adadelta/conv2d_19/kernel/accum_var
-:+ 2!Adadelta/conv2d_19/bias/accum_var
;:9 2#Adadelta/conv2d_20/kernel/accum_var
-:+2!Adadelta/conv2d_20/bias/accum_var
î2ë
 __inference__wrapped_model_25130Æ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *6¢3
1.
conv2d_14_inputÿÿÿÿÿÿÿÿÿ22
þ2û
,__inference_sequential_2_layer_call_fn_25775
,__inference_sequential_2_layer_call_fn_25808
,__inference_sequential_2_layer_call_fn_25597
,__inference_sequential_2_layer_call_fn_25519À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_25690
G__inference_sequential_2_layer_call_and_return_conditional_losses_25742
G__inference_sequential_2_layer_call_and_return_conditional_losses_25395
G__inference_sequential_2_layer_call_and_return_conditional_losses_25440À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_14_layer_call_fn_25827¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_14_layer_call_and_return_conditional_losses_25818¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_12_layer_call_fn_25837¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_12_layer_call_and_return_conditional_losses_25832¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_15_layer_call_fn_25856¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_15_layer_call_and_return_conditional_losses_25847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_13_layer_call_fn_25866¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_13_layer_call_and_return_conditional_losses_25861¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_16_layer_call_fn_25885¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_16_layer_call_and_return_conditional_losses_25876¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_14_layer_call_fn_25895¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_14_layer_call_and_return_conditional_losses_25890¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_17_layer_call_fn_25914¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_17_layer_call_and_return_conditional_losses_25905¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_15_layer_call_fn_25924¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_15_layer_call_and_return_conditional_losses_25919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_18_layer_call_fn_25943¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_18_layer_call_and_return_conditional_losses_25934¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_16_layer_call_fn_25953¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_16_layer_call_and_return_conditional_losses_25948¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_19_layer_call_fn_25972¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_19_layer_call_and_return_conditional_losses_25963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_17_layer_call_fn_25982¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_17_layer_call_and_return_conditional_losses_25977¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_20_layer_call_fn_26001¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_20_layer_call_and_return_conditional_losses_25992¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:B8
#__inference_signature_wrapper_25638conv2d_14_input¶
 __inference__wrapped_model_25130()23<=FGPQ@¢=
6¢3
1.
conv2d_14_inputÿÿÿÿÿÿÿÿÿ22
ª "=ª:
8
	conv2d_20+(
	conv2d_20ÿÿÿÿÿÿÿÿÿ22´
H__inference_activation_12_layer_call_and_return_conditional_losses_25832h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
-__inference_activation_12_layer_call_fn_25837[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
H__inference_activation_13_layer_call_and_return_conditional_losses_25861h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
-__inference_activation_13_layer_call_fn_25866[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
H__inference_activation_14_layer_call_and_return_conditional_losses_25890h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
-__inference_activation_14_layer_call_fn_25895[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
H__inference_activation_15_layer_call_and_return_conditional_losses_25919h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
-__inference_activation_15_layer_call_fn_25924[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
H__inference_activation_16_layer_call_and_return_conditional_losses_25948h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
-__inference_activation_16_layer_call_fn_25953[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
H__inference_activation_17_layer_call_and_return_conditional_losses_25977h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
-__inference_activation_17_layer_call_fn_25982[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
D__inference_conv2d_14_layer_call_and_return_conditional_losses_25818l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
)__inference_conv2d_14_layer_call_fn_25827_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª " ÿÿÿÿÿÿÿÿÿ22 ´
D__inference_conv2d_15_layer_call_and_return_conditional_losses_25847l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
)__inference_conv2d_15_layer_call_fn_25856_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
D__inference_conv2d_16_layer_call_and_return_conditional_losses_25876l()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
)__inference_conv2d_16_layer_call_fn_25885_()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
D__inference_conv2d_17_layer_call_and_return_conditional_losses_25905l237¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
)__inference_conv2d_17_layer_call_fn_25914_237¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
D__inference_conv2d_18_layer_call_and_return_conditional_losses_25934l<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
)__inference_conv2d_18_layer_call_fn_25943_<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
D__inference_conv2d_19_layer_call_and_return_conditional_losses_25963lFG7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22 
 
)__inference_conv2d_19_layer_call_fn_25972_FG7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22 ´
D__inference_conv2d_20_layer_call_and_return_conditional_losses_25992lPQ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
)__inference_conv2d_20_layer_call_fn_26001_PQ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22 
ª " ÿÿÿÿÿÿÿÿÿ22Õ
G__inference_sequential_2_layer_call_and_return_conditional_losses_25395()23<=FGPQH¢E
>¢;
1.
conv2d_14_inputÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 Õ
G__inference_sequential_2_layer_call_and_return_conditional_losses_25440()23<=FGPQH¢E
>¢;
1.
conv2d_14_inputÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 Ì
G__inference_sequential_2_layer_call_and_return_conditional_losses_25690()23<=FGPQ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 Ì
G__inference_sequential_2_layer_call_and_return_conditional_losses_25742()23<=FGPQ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 ¬
,__inference_sequential_2_layer_call_fn_25519|()23<=FGPQH¢E
>¢;
1.
conv2d_14_inputÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22¬
,__inference_sequential_2_layer_call_fn_25597|()23<=FGPQH¢E
>¢;
1.
conv2d_14_inputÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22£
,__inference_sequential_2_layer_call_fn_25775s()23<=FGPQ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22£
,__inference_sequential_2_layer_call_fn_25808s()23<=FGPQ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22Ì
#__inference_signature_wrapper_25638¤()23<=FGPQS¢P
¢ 
IªF
D
conv2d_14_input1.
conv2d_14_inputÿÿÿÿÿÿÿÿÿ22"=ª:
8
	conv2d_20+(
	conv2d_20ÿÿÿÿÿÿÿÿÿ22