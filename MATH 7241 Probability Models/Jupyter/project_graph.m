P = [0.8 0.2 0 0;
     0 0 0.5 0.5;
     0.5 0.5 0 0;
     0 0 0.5 0.5];
 
stateNames = {'SS','SF','FS','FF'};

mc = dtmc(P,'StateNames',stateNames);
figure;
graphplot(mc,'ColorEdges',true);
title('Network Graph of Markov Chain, Exercise 3')
classify(mc)
asymptotics(mc)

