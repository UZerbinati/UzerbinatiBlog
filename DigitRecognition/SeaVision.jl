using Flux, Statistics
using Flux.Tracker: update!, grad, back!
using Flux: onehotbatch, onecold, crossentropy, throttle
using Random
using InformationMeasures
using Distributions
using Base.Iterators: repeated, partition
using LinearAlgebra
using Plots
using TickTock

p=0.7

function ReadSEAVISION(filename)
    f = open(filename)
    A = map(row -> map(x -> parse(Int, x), split(row, ',')), eachline(f))
    close(f)

    T = [a[2:end] for a in A]
    L = [a[1] for a in A]
	N = length(A[1])

	T = reduce(hcat, T)

    return length(A), N-1, T, L
end
function ReadMNIST(filename)
    f = open(filename)
    A = map(row -> map(x -> parse(Int, x), split(row, ',')), eachline(f))
    close(f)

    T = [a[2:end] for a in A]
    L = [a[1] for a in A]
	N = length(A[1])
    return length(A), Int(sqrt(N-1)), T, L
end
function MakeData(N, K)

	M = N/K

	Ls = Array{Float64,1}[]

	for k = 1:K

		for i = 1:M

			push!(Ls, [k,k]+0.5*randn(2) )

		end

	end

	return Ls
end
function RandomAssign(X, k)

	unif = DiscreteUniform(1, k)

	return [rand(unif) for x in X]
end
Partition(X, C) = [[X[i] for i=1:length(X) if C[i] == j] for j=1:maximum(C)]
function zOpt(X,C,k)
        G=Partition(X,C);
	#print(C);
       	#Cerchiamo ora il vettore z
	#print(G[1]);
	#print(length(G[1]));
	Z=[];
	for i in 1:k
		z = zeros(1491,1);
		#println("Dimensioni",length(G[i]));
		for j in 1:length(G[i])
			z=z+G[i][j];

		end
		z = (1/length(G[i]))*z;
		Z = push!(Z,z);
		#println(z);
	end
	return(Z);
end
#Volgio ora ricpmputare il vettore C
function kOpt(X,Z,k)
	K = [];
	for i in 1:length(X)
		D = [];
		for j in 1:k
			push!(D,norm(X[i]-Z[j]));
		end
		push!(K,findmin(D)[2]);
	end
	return(K);
end
#Computing optimum function
function vOpt(C,X,Z)
	v=0;
	for i in 1:length(X)
		j = C[i];
		v = v+ norm([X[i][1]-Z[j][1],X[i][2]-Z[j][2]])
	end
	v = (1/length(X))*v;
	return(v);
end
function AnalysisTool(X,YB,Y,n,NN)
    NX = onecold(NN(X));
    NY = onecold(YB);
    PX = NN(X)
    E = [];
    EX = [];
    println(n)
    for i in 1:n
        if NX[i] != NY[i]
            println("Errore Nel Riconoscere La Digita: ",Y[i])
            push!(EX,X[:,i]);
            push!(E,PX[:,i]);
        end
    end
    #println(E)
    fig1 = display(PlotDigit(EX,length(EX),1));
    println(mean([var(PX[:,i]) for i in 1:n]))
    println(mean([var(e) for e in E]))
    k=0;
    for i in 1:n
        if var(PX[:,i])< 0.032
            k=k+1;
        end
    end
    println("Number of poit with dout: ",k);
end
function AnalysisToolBox(X,Y,n,l,NN,Z)
    YB=onehotbatch(Y,0:9)
    NX = onecold(NN(X));
    NY = onecold(YB);
    PX = NN(X)
    EP = [];
    EN =[];
    EX = [];
    println(n)
    k=0;
    kk=0;
    kkk=0;
    kkkk=0;
    for i in 1:n
        if NX[i] != NY[i]
            push!(EN,Y[i]);
            push!(EX,X[:,i]);
            push!(EP,PX[:,i]);
            kkk=kkk+1;
            if maximum(PX[:,i]) > p
                kkkk=kkkk+1;
            end
        end
        if maximum(PX[:,i]) < p
            k=k+1
            v,a = findmin([norm(X[:,i]-Z[j]) for j in 1:l]);
            #println("test",onecold(NN(Z[a])),NY[i])
            if onecold(NN(Z[a]))[1] != NY[i]
                 kk=kk+1;
            end
        end
    end
    #println("Entropy Media DataSet: ", mean([get_entropy(Float32(PX[:,i])) for i in 1:n]))
    for i in 1:length(EP)
        #println("Digit: ",EN[i]," ",var(EP[i]),[(onecold(NN(Z[j])),norm(EX[i]-Z[j])) for j in 1:l],EP[i])
        #println("Digit: ",EN[i]," ",var(EP[i]),[(onecold(NN(Z[j])),norm(EX[i]-Z[j],1)) for j in 1:l],EP[i])
        #println("Digit: ",EN[i]," ",var(EP[i]),[(onecold(NN(Z[j])),norm(EX[i]-Z[j],Inf)) for j in 1:l],EP[i])
        v,k1 = findmin([norm(EX[i]-Z[j]) for j in 1:l]);
        v,k2 = findmin([norm(EX[i]-Z[j],1) for j in 1:l]);
        v,k3 = findmin([norm(EX[i]-Z[j],Inf) for j in 1:l]);
        println("Digit: ", EN[i]," Variance: ",var(EP[i])," Digita prevista da baricentro: ", onecold(NN(Z[k1])),onecold(NN(Z[k2])),onecold(NN(Z[k3])));
        println(EP[i]);
    end
    println("Varianza Media DataSet: ", mean([var(PX[:,i]) for i in 1:n]))
    println("Digite su cui sono in dubbio: ",k," tra queste sono sbagliati: ",kk, " Cifre Sbagliate totale: ",kkk, " tra queste quelle non dubbie: ", kkkk);

end
function PlotDigits(row, k, w)
	R(X) = reshape(X, (71,21))[end:-1:1,:]

	A = zeros(71*w, 21*k)
	idx = 0
	for j = 1:w
		for i = 1:k
			idx = idx + 1
			A[(1+71*(j-1)):(71*j), (21*(i-1)+1):(21*(i))] = R(row[:,idx])
		end
	end

    p = heatmap(A, subplot=1, aspect_ratio=1, xticks = nothing, yticks = nothing,
    	         colorbar=false, c=cgrad([:red, :blue, :white]))

    #png(p, "digits")
end
function PlotDigit(row, k, w)
	R(X) = reshape(X, (71,21))[end:-1:1,:]

	A = zeros(71*w, 21*k)
	idx = 0
	for j = 1:w
		for i = 1:k
			idx = idx + 1
			A[(1+71*(j-1)):(71*j), (21*(i-1)+1):(21*(i))] = R(row[idx])
		end
	end

    p = heatmap(A, subplot=1, aspect_ratio=1, xticks = nothing, yticks = nothing,
    	         colorbar=false, c=cgrad([:red, :blue, :white]))

    #png(p, "digits")
end


#In questa funzione costruisco la rete
function BuildNN(it)
    # Legge i dati di training
    n, m, X, Y = ReadSEAVISION("sv.csv")
    YB=onehotbatch(Y,0:9) #Trasformo in matrice di valori Booleani
    # Stampa le prime 20 digits
    #display(PlotDigits(X, 10, 2))
    prediction = Chain(
      Dense(1491, 200,sigmoid),
      Dense(200,10),softmax) |> gpu
    # Loss function: crossentropy
    L(x, y) = crossentropy(prediction(x), y)
    dataset = repeated((X, YB),it)
    opt = ADAM(); #ADAM (Slide Reti Multilayer)
    callback = () -> @show(L(X, YB)) #Mostro valore della funzione loss ad ogni Training
    Flux.train!(L, Flux.params(prediction), dataset, opt,cb=callback) #Faccio il Training

    acc(x, y) = mean(onecold(prediction(x)) .== onecold(y)) #Valuto l' accurtezza.
    println("Accurtezza:",acc(X,YB))
    #AnalysisTool(X,YB,Y,n,prediction)
    return  prediction
end
#In questa funzioni faccio una KMEANS
function KMeansClauster1()
    Random.seed!(13)
	maxit=300;


	k = 15

	#X = MakeData(300, k)
	n, m, I, Y = ReadSEAVISION("sv.csv");
    X = [I[:,i] for i in 1:n]
	#print(length(X[1]));
	C = RandomAssign(X, k)
	v=[];
	Z=[];
	for i in 1:maxit
		Z = zOpt(X,C,k)
        #println("Z",length(Z))
		#v=push!(v,vOpt(C,X,Z));
		C = kOpt(X,Z,k)
		#if ((v[i-1]-v[i])<0.0000000002)
		#	break;
		#end
	end
    #fig2 = display(PlotDigit(Z,15,1))
    return Z,C
end
function KMeansClauster2()
    Random.seed!(13)
	maxit=300;


	k = 2

	#X = MakeData(300, k)
	n, m, I, Y = ReadSEAVISION("5_6.csv");
    X = [I[:,i] for i in 1:n];
	#print(length(X[1]));
	C = RandomAssign(X, k)
	v=[];
	Z=[];
	for i in 1:maxit
		Z = zOpt(X,C,k)
        #println("Z",length(Z))
		#v=push!(v,vOpt(C,X,Z));
		C = kOpt(X,Z,k)
		#if ((v[i-1]-v[i])<0.0000000002)
		#	break;
		#end
	end
    #fig2 = display(PlotDigit(Z,2,1))
    return Z,C
end

tick();

NN=BuildNN(3750);
Z, C = KMeansClauster1();
ZZ, CC= KMeansClauster2();
CY = [];
deleteat!(Z,3);
deleteat!(Z,4);
deleteat!(Z,5);
deleteat!(Z,9);
push!(Z,ZZ[1]);
println(length(Z))
l=12
for k in 1:l
    println("NN su Baricentri: ",NN(Z[k]));
end


function Accuratezza(f,X,Y,fn)
    D=[];
    io = open(fn,"a+");
    k=0.0;
    n = size(X)[2];
    write(io,"Risultati Rete NN\n");
    for i in 1:n
        #println(f(X[:,i]));
        if (f(X[:,i])-1) == Y[i]
            k=k+1.0;
        else
            println("Errore nela ciifra: ", i);
            push!(D,X[:,i]);
        end
        #write(io,String(f(X[:,i])-1));
        write(io,string(f(X[:,i])-1));
        write(io,"\n");
    end
    return (k/n),D;
end

function DigitRecognition(X)
    Y = NN(X);
    #println("M: ",maximum(Y));
    #Applico un opzione di selta individuata studiando i dati con l' AnalysisToolBox
    if maximum(Y) < p
        v,a = findmin([norm(X-Z[j]) for j in 1:l]);
        F = onecold(NN(Z[a]))[1];
        println("Decision Based Upon Bariceter")
    else
        F = onecold(Y);
    end
    return F;
end

function testSV()
    n2,m2,X2,A2=ReadSEAVISION("sv.csv");
    n,m,X,A=ReadSEAVISION("SVT.csv");
    Y=[5, 4, 9, 7, 4, 3, 4, 5, 0, 4, 4, 3, 8, 0, 3, 4, 4, 4, 1, 4, 1, 5, 2, 1, 1, 0, 0, 1, 1, 1, 1, 4, 3, 2, 0, 4, 1, 0, 1, 4, 8, 0, 1, 4, 0, 0, 9, 0, 4, 9, 1, 1, 9, 7, 1, 4, 1, 1, 2, 0, 1, 4, 9, 1, 1, 5, 1, 3, 1, 9, 0, 5, 4, 5, 9, 9, 5, 9, 6, 1, 5, 0, 5, 9, 0, 5, 5, 1, 4, 6, 1, 0, 4, 1, 0, 1, 4, 0, 1, 1, 0, 9, 0, 4, 0, 2, 4, 1, 1, 9, 4, 5, 9, 9, 4, 9, 9, 5, 4, 4, 5, 4, 2, 1, 1, 5, 4, 1, 3, 8, 1, 0, 5, 9, 2, 2, 8, 9, 5, 8, 3, 5, 9, 1, 9, 4, 3, 1, 0, 5, 5, 0, 5, 9, 1, 1, 0, 7, 1, 0, 3, 1, 5, 1, 0, 3, 5, 4, 5, 1, 4, 9, 4, 4, 0, 5, 5, 4, 4, 5, 4, 0, 5, 1, 0, 5, 3, 9, 7, 5, 4, 2, 1, 9, 1, 4, 5, 9, 1, 2, 9, 5, 4, 5, 0, 6, 8, 0, 5, 0, 5, 1, 6, 5, 1, 5, 9, 0, 1, 5, 0, 9, 1, 5, 6, 5, 4, 7, 4, 3, 2, 9, 1, 5, 1, 1, 0, 0, 5, 1, 5, 0, 0, 5, 3, 8, 4, 4, 4, 5, 0, 1, 5, 8, 5, 5, 1, 1, 1, 4, 1, 9, 4, 2, 1, 8, 4, 1, 9, 7, 5, 1, 4, 1, 1, 5, 2, 4, 9, 3, 4, 0, 6, 1, 3, 1, 0, 1, 1, 1, 0, 6, 5, 5, 4, 9, 4, 0, 9, 1, 0, 9, 1, 4, 5, 3, 5, 1, 4, 1, 0, 1, 4, 8, 9, 0, 1, 4, 4, 1, 4, 4, 0, 0, 5, 9, 9, 4, 8, 4, 2, 4, 0, 4, 9, 5, 5, 5, 0, 5, 1, 5, 9, 0, 0, 5, 0, 0, 4, 5, 5, 2, 0, 0, 9, 0, 1, 2, 5, 0, 0, 1, 6, 1, 4, 1, 5, 4, 9, 5, 5, 9, 4, 0, 5, 3, 1, 0, 5, 0, 5, 4, 5, 5, 4, 1, 7, 4, 9, 0, 8, 4, 5, 5, 5, 5, 4, 1, 8, 5, 0, 5, 4, 2, 9, 1, 0, 4, 0, 5, 1, 0, 5, 1, 5, 8, 1, 5, 1, 4, 7, 4, 1, 0, 4, 7, 9, 9, 0, 9, 1, 5, 4, 9, 5, 8, 1, 1, 9, 4, 5, 0, 5, 2, 1, 4, 0, 1, 9, 4, 4, 7, 5, 4, 1, 9, 0, 5, 0, 1, 1, 4, 9, 1, 1, 1, 6, 9, 1, 5, 1, 5, 0, 9, 1, 5, 3, 9, 1, 0, 8, 9, 9, 4, 6, 5, 5, 0, 4, 5, 5, 2, 0, 5, 9, 2, 0, 5, 0, 4, 3, 1, 8, 2, 9, 1, 6, 0, 9, 1, 5, 1, 0, 5, 3, 3, 5, 3, 1, 3, 4, 1, 4, 1, 1, 4, 0, 1, 1, 1, 5, 1, 4, 4, 2, 5, 5, 5, 1, 7, 5, 8, 0, 4, 0, 9, 9, 5, 5, 1, 5, 1, 1, 2, 5, 1, 4, 5, 1, 4, 1, 3, 1, 3, 4, 4, 6, 9, 5, 3, 5, 5, 0, 8, 9, 0, 0, 5, 0, 4, 1, 1, 4, 0, 0, 6, 5, 4, 0, 3, 0, 1, 4, 9, 0, 5, 3, 1, 5, 3, 2, 4, 4, 0, 1, 2, 1, 0, 4, 8, 4, 1, 1, 0, 5, 5, 1, 6, 1, 4, 5, 1, 1, 9, 0, 0, 6, 4, 8, 4, 8, 6, 5, 0, 8, 9, 7, 1, 0, 1, 0, 5, 5, 5, 4, 9, 0, 9, 4, 0, 0, 4, 9, 0, 4, 5, 0, 0, 4, 0, 0, 5, 5, 2, 9, 4, 0, 1, 9, 8, 5, 1, 0, 3, 9, 1, 5, 1, 3, 8, 3, 0, 5, 4, 1, 9, 0, 1, 5, 5, 8, 4, 1, 4, 6, 0, 5, 0, 1, 9, 4, 0, 1, 9, 1, 0, 9, 1, 1, 0, 8, 0, 6, 5, 9, 4, 4, 1, 9, 8, 5, 4, 1, 1, 1, 1, 1, 9, 4, 5, 4, 1, 5, 1, 9, 5, 9, 1, 4, 3, 5, 5, 4, 5, 9, 5, 0, 5, 4, 2, 9, 8, 3, 8, 9, 5, 4, 0, 4, 9, 4, 9, 5, 5, 9, 9, 2, 0, 3, 1, 1, 4, 5, 4, 6, 1, 0, 7, 5, 9, 7, 5, 0, 5, 6, 5, 4, 0, 3, 4, 1, 2, 5, 1, 0, 0, 0, 9, 4, 5, 5, 6, 6, 0, 0, 5, 2, 0, 8, 5, 6, 6, 0, 9, 4, 8, 4, 1, 1, 4, 5, 3, 1, 1, 1, 9, 7, 3, 5, 9, 1, 4, 0, 5, 6, 4, 0, 9, 9, 5, 0, 1, 9, 0, 5, 2, 0, 2, 9, 9, 3, 5, 0, 9, 1, 0, 1, 9, 5, 6, 5, 4, 9, 0, 1, 4, 6, 4, 9, 1, 5, 5, 1, 9, 7, 1, 4, 9, 4, 2, 0, 0, 6, 1, 9, 9, 4, 5, 4, 8, 1, 5, 9, 5, 9, 1, 8, 1, 0, 3, 5, 4, 0, 9, 0, 1, 4, 1, 0, 7, 9, 9, 0, 9, 1, 5, 1, 4, 4, 1, 5, 2, 4, 5, 7, 9, 5, 9, 4, 1, 9, 0, 5, 4, 0, 1, 3, 0, 4, 5, 4, 2, 0, 4, 1, 5, 0, 0, 8, 4, 9, 5, 9, 1, 0, 4, 4, 9, 1, 0, 1, 0, 0, 0, 6, 5, 6, 1, 1, 5, 1, 5, 9, 5, 5, 0, 0, 5, 1, 0, 9, 4, 5, 4, 4, 4, 0, 1, 6, 4, 4, 0, 7, 5, 1, 5, 2, 4, 9, 0, 1, 5, 1, 5, 4, 0, 5, 1, 1, 1, 9, 3, 5, 4, 0, 5, 0, 7, 5, 5, 2, 0, 1, 4, 0, 4, 5, 2, 1, 5, 9, 1, 4, 1, 5, 5, 9, 5, 4, 1, 6, 5, 4, 0, 8, 0, 3, 0, 0, 1, 0, 7, 5, 0, 4, 7, 1, 3, 4, 0, 8, 2, 5, 5, 0, 8, 5, 0, 4, 0, 1, 1, 0, 6, 1, 5, 5, 1, 1, 0, 5, 4, 9, 8, 0, 0, 0, 1, 1, 5, 2, 0, 5, 5, 4, 2, 0, 5, 0, 0, 2, 4, 4, 6, 3, 5, 1, 2, 5, 9, 0, 9, 5, 4, 4, 5, 1, 5, 0, 1, 1, 4, 9, 7, 4, 4, 0, 4, 9, 1, 6, 4, 1, 1, 0, 0, 0, 3, 9, 5, 0, 4, 5, 4, 8, 3, 5, 6, 1, 8, 1, 7, 0, 0, 0, 1, 5, 5, 9, 2, 0, 9, 5, 9, 1, 4, 5, 1, 0, 9, 4, 1, 2, 1, 0, 5, 9, 0, 6, 2, 0, 5, 4, 0, 9, 5, 1, 5, 5, 5, 1, 5, 5, 5, 0, 0, 4, 6, 4, 8, 1, 4, 4, 1, 4, 4, 9, 0, 1, 5, 4, 4, 9, 9, 5, 0, 9, 5, 2, 3, 4, 0, 8, 3, 0, 1, 2, 0, 2, 9, 1, 1, 4, 2, 5, 5, 9, 1, 0, 5, 4, 9, 9, 8, 4, 1, 0, 9, 3, 0, 0, 5, 5, 1, 0, 8, 5, 0, 9, 0, 1, 9, 7, 4, 5, 0, 5, 6, 5, 4, 1, 9, 0, 0, 5, 1, 3, 5, 5, 8, 9, 5, 0, 0, 1, 5, 1, 5, 4, 9, 9, 5, 1, 5, 9, 6, 9, 4, 0, 0, 8, 1, 9, 6, 1, 5, 5, 4, 5, 1, 7, 0, 9, 8, 9, 5, 0, 4, 1, 9, 5, 4, 1, 1, 4, 0, 8, 9, 4, 1, 1, 5, 1, 2, 0, 2, 5, 5, 9, 1, 0, 5, 9, 1, 0, 5, 6, 5, 0, 3, 9, 4, 1, 0, 8, 4, 0, 9, 7, 1, 9, 1, 0, 1, 4, 9, 4, 1, 4, 5, 4, 1, 4, 4, 1, 0, 1, 9, 5];
    Y[635]=6;
    Y[641]=8;
    Y[705]=0;
    Y[1157]=1;
    Y[1367]=4;
    YB=onehotbatch(Y,0:9);
    AB=onehotbatch(A2,0:9);
    #AnalysisToolBox(X,Y,n,l,NN,Z);
    perc, D = Accuratezza(DigitRecognition,X,Y,"SVR.txt");
    perc2, D2 = Accuratezza(DigitRecognition,X2,A2,"SVTR.txt");
    println("Accurtezza Test Set: ",perc);
    println("Accurtezza Data Set:", perc2);
    display(PlotDigit(D2,length(D2),1));
    return(X,Y)
end

X,Y = testSV()
tock();
