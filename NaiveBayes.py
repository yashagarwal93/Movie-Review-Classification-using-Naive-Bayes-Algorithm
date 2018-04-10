import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
import math
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.X.shape[1]
        self.count_positive = np.zeros([1,data.X.shape[1]])
        self.count_negative = np.zeros([1,data.X.shape[1]])
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.0
        self.P_negative = 0.0
        self.deno_pos = 0.0
        self.deno_neg = 0.0
        self.precision = 0
        self.recall = 0
        
        self.num_hit = 0
        self.num_target = 0
        self.Train(data.X,data.Y)
           

    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        
        self.num_positive_reviews = len(positive_indices)
        self.num_negative_reviews = len(negative_indices)
        
        self.count_P = np.ix_(positive_indices)
        self.count_N = np.ix_(negative_indices)
        
        self.count_positive = self.count_positive + csr_matrix.sum(X[self.count_P], axis=0) + self.ALPHA
        self.count_negative = self.count_negative + csr_matrix.sum(X[self.count_N], axis=0) + self.ALPHA
        
        self.total_positive_words = csr_matrix.sum(X[self.count_P])
        self.total_negative_words = csr_matrix.sum(X[self.count_N])
        
        self.deno_pos = float(self.total_positive_words + self.ALPHA * X.shape[1])
        self.deno_neg = float(self.total_negative_words + self.ALPHA * X.shape[1])
        
            
        return


    def PredictLabel(self, X):
        #TODO: Implement Naive Bayes Classification
        
               
        P_review= float(self.num_positive_reviews)
        N_review= float(self.num_negative_reviews)
        P_review= math.log(P_review)
        N_review= math.log(N_review)
        self.P_positive = P_review - (P_review + N_review)
        self.P_negative = N_review - (P_review + N_review)
        pred_labels = []
        
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            P_positive_c= self.P_positive
            P_negative_c=self.P_negative            
            for j in range(len(z[0])):
                #col = z[1][j]
                x = X[i, z[1][j]]
                p = self.count_positive[0, z[1][j]]
                n = self.count_negative[0, z[1][j]]
                positive_score = math.log(p) - math.log(self.deno_pos)
                negative_score = math.log(n) - math.log(self.deno_neg)
                P_positive_c = P_positive_c + x * positive_score         
                P_negative_c = P_negative_c + x * negative_score
                
                            
            if P_positive_c > P_negative_c:
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)
        
        return pred_labels
       
    def PredictLabel_threshold(self, X,probThresh):
        #TODO: Implement Naive Bayes Classification
        
        P_review= float(self.num_positive_reviews)
        N_review= float(self.num_negative_reviews)
        P_review= math.log(P_review)
        N_review= math.log(N_review)
        self.P_positive = P_review - (P_review + N_review)
        self.P_negative = N_review - (P_review + N_review)
        
        pred_labels = []
        
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            P_positive_c= self.P_positive
            P_negative_c=self.P_negative              
            for j in range(len(z[0])):
                #col = z[1][j]
                x = X[i, z[1][j]]
                p = self.count_positive[0, z[1][j]]
                n = self.count_negative[0, z[1][j]]
                positive_score = math.log(p) - math.log(self.deno_pos)
                negative_score = math.log(n) - math.log(self.deno_neg)
                P_positive_c = P_positive_c + x * positive_score         
                P_negative_c = P_negative_c + x * negative_score
                #percentage = P_positive_c / (P_positive_c + P_negative_c)
                percentage = exp(P_positive_c - self.LogSum(P_positive_c,P_negative_c))
                
            
            
            if percentage > probThresh:
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)
        
        return pred_labels
        #return 1
    
    def LogSum(self, logx, logy):   
        # TO Do: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    def PredictProb(self, test, indexes):
    
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            predicted_label = 0
            z = test.X[i].nonzero()
            P_positive_c= self.P_positive
            P_negative_c=self.P_negative 
            for j in range(len(z[0])):
                x = test.X[i, z[1][j]]        
                p = math.log(self.count_positive[0, z[1][j]]) 
                n = math.log(self.count_negative[0, z[1][j]])
                P_positive_c = P_positive_c + x * p               
                P_negative_c = P_negative_c + x * n
            
            #predicted_prob_positive = 0.5
            #predicted_prob_negative = 0.5
            predicted_prob_positive = exp(P_positive_c - self.LogSum(P_positive_c, P_negative_c))
            predicted_prob_negative = exp(P_negative_c - self.LogSum(P_positive_c, P_negative_c))
            
            sum_positive=P_positive_c
            sum_negative=P_negative_c
            
            if sum_positive > sum_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print (test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
        
        # For plotting the graph against precision and recall
        l= (0.22,0.32,0.42,0.52,0.62)
        precision_list=[]
        recall_list=[]
        
        for i in range(len(l)):            
            Y_pred_1 = self.PredictLabel_threshold(test.X,l[i])
            #ev = Eval(Y_pred_1, test.Y)
            #Y_pred1 = self.PredictProb(test.X)
        
            precision_prob = self.EvalPrecision(Y_pred_1,test.Y)
            recall_prob = self.EvalRecall(Y_pred_1,test.Y)
            #print ("Precision:",precision_label)
            #print ("Recall:",recall_label)
            print ("Precision:",precision_prob)
            print ("Recall:",recall_prob)
            precision_list.append(precision_prob)
            recall_list.append(recall_prob)
        plt.plot(precision_list,recall_list)
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.savefig("graph.png")
        #print (Y_pred, Y_gold)
           
      
    
    def EvalPrecision(self, target, predict):
        
        tp=0
        fp=0
        for i in range(len(target)):
            if (target[i] ==1 and predict[i] == 1):
                tp= tp+1
            elif (target[i] ==1 and predict[i] == -1):
                fp = fp+1
        self.precision= tp/(tp+fp)
        return self.precision
    
    def EvalRecall(self, target, predict):
        
        fn=0
        tn=0
        tp=0
        for i in range(len(target)):
            if (target[i] == -1 and predict[i] == 1):
                fn= fn+1
            elif (target[i] == -1 and predict[i] == -1):
                tn = tn+1
            elif (target[i] == 1 and predict[i] == 1):
                tp= tp+1
        self.recall= tp/(tp+fn)
        return self.recall
                     
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()
    
    def getWords(self, X, vocab):
        self.P_positive = log(float(self.num_positive_reviews)) - log(float(self.num_positive_reviews + self.num_negative_reviews))
        self.P_negative = log(float(self.num_negative_reviews)) - log(float(self.num_positive_reviews + self.num_negative_reviews))
        ft_wt_pos = {}
        ft_wt_neg = {}
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            sum_positive = self.P_positive
            sum_negative = self.P_negative
            for j in range(len(z[0])):
                row_index = i
                col_index = z[1][j]
                times = X[row_index, col_index]
                
        
                P_pos = log(self.count_positive[0, col_index]) - log(self.deno_pos)
                
                sum_positive = sum_positive + times * P_pos
                
                P_neg = log(self.count_negative[0, col_index]) - log(self.deno_neg)
                sum_negative = sum_negative + times * P_neg
                word = vocab.GetWord(int(col_index))
                if word not in ft_wt_pos:
                    ft_wt_pos[word] = P_pos
                else:
                    ft_wt_pos[word] += P_pos 
                if word not in ft_wt_neg:
                    ft_wt_neg[word] = P_neg
                else:
                    ft_wt_neg[word] += P_neg  
            
         
        twenty_largest_p = sorted(ft_wt_pos, key=ft_wt_pos.get)[:100]
        twenty_largest_n = sorted(ft_wt_neg, key=ft_wt_neg.get)[:100]
       
        print("Top 20 Positive Words: ")
        for i in twenty_largest_p:
            print(i, ft_wt_pos[i])
        print("Top 20 Negative Words: ")
        for i in twenty_largest_n:
            print(i, ft_wt_neg[i])


if __name__ == "__main__":
    #data = IMDBdata("../data/aclImdb")
    print ("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print ("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    print ("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print ("Evaluating")
    print ("Test Accuracy: ", nb.Eval(testdata))
    print (nb.PredictProb(testdata, range(10)))  # predict the probabilities of reviews being positive (first 10 reviews in the test set)
    print (nb.getWords(traindata.vocab))
    
