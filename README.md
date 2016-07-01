# Headline-Generation 

Recent advances in deep learning have lead to an explosion of research in the field, both applied and theoretical. Research has focused on a breadth of different areas, with large amounts narrowing in on image and text processing. A [fascinating paper](http://nlp.stanford.edu/courses/cs224n/2015/reports/1.pdf) that I have recently read attempts to predict headlines of news articles using the article body text as input. 

This paper has applications for *text summarization* across any body of text, and as such I wanted to dive in! 

## Early Stage Results 

At this point, I have gotten some pretty encouraging results. After working through a couple of smaller networks and making sure that I could build a recurrent network architecture that would learn to overfit the data, I iteratively built up a larger model and tuned it to generalize as well as possible to a validation set (the current best architecture is located in `headline-generation/model/model.py`). 

In terms of preliminary results, below I've shown 5 examples from the training and validation sets. Each includes the raw text (portion of article body) fed into the recurrent network, the true headline observed in the data, and the predicted headline from the network. Note that the text seems a little nonsensical, which is because stopwords and punctuation were removed before feeding it into the network.  

#### Validation Set Examples

| | Raw Text (article body) | True Headline | Predicted Headline | 
|---|-------------------------|---------------|--------------------|
|1| article wayne smith writes article richard krehbiel writes anyone explain fairly simple terms get might need scsi rather ide performance suffer much drive dont tape drive cdrom help would appreciated got multitasking want increase performance increasing amount overlapping one way dma bus mastering either make possible io devices move | ide vs scsi dma detach | ide vs scsi |    
|2| belongs ive redirected followups article larry snyder writes xfree86 support eisa video cards dell 22 larry snyder know fact eisa version orchid iis works however eisa svga card likely waste money xfree86 20 comes support accelerated chipsets isa eisa vlb supported important question chipsets supported bus basically irrelevent compatibility | dell 22 eisa video cards | video card | 
|3| need help multi port serial board unknown origin im hoping someone knows board even better various switches used anyway heres description card card although noticed none contacts extension connected anything 4 chips sockets 4 corresponding labeled also external female connector 37 pins 8 banks 8 switches 2 banks 4 | need help identifying serial board | help making io mouse |   
|4| article mike writes im quite familiar variety window title setting methods question way via resources etc stop applications ability rename name properties sorry thats feature specifies app set title wm obliged bothers complain app writer cares nice application want control write wm doesnt support write program give window id | way stop application retitling | forcing window manager accept specific coordinates window |  
|5| message wed 28 apr 1993 gmt initial references best regards walter author walid jahir pabon robert young title integrating parametric geometry features variational modeling conceptual design international conference design theory methodology year 1990 editor j r pages 19 organization american society mechanical engineers asme note proceedings author yasushi yamaguchi | design |scientific cult |  

These five examples exemplify the three general kinds of examples I saw when looking at predictions on the validation set. The first two (`1` and `2`) are examples where the net is accurately predicting part of the headline, but not all of it. With examples `3` and `4`, the net is in the right ballpark (e.g. the topic of the predicted headline seems to line up at least somewhat with the topic of the true headline), but isn't quite capturing the content of the raw text that's fed in. The last one, `5`, isn't really close (and is all also just kind of funny). Maybe the prediction of the word `scientific` makes sense, but I also think that might be a stretch, especially with the word `cult` after it.  

#### Training Examples 

| | Raw Text (article body) | True Headline | Predicted Headline | 
|----|-----------|--------------|-------|
|1| distribution world tobias doping writes try display window hints tell window manager position size window specified users window manager accept values use tells window manager values prefered values program user dont know window manager doesnt place window like prefer specify position size like sorry dont place title position window | forcing window manager accept specific coordinates window | forcing window manager accept specific coordinates window |   
|2| distribution na article scott w roby writes find disturbing good keep thinking critically dont patronize wont patronize feel free patronize like need tips seriously insulted apologize tiresome thing group many people tell others sucking government ever decide something government says plausible praise independent thinkers whenever find something government says | photographers removed compound | photographers removed compound | 
|3| hey man spent past season learning skate played couple sessions mock hockey im ready invest hockey equipment particularly since taking summer hockey lessons however completely profoundly ignorant comes hockey equipment ive checked local stores looked catalogs hoping solicit actually plunking money played football high school college least equipment basis | hockey equip recommendations | hockey equip recommendations |   
|4| got dot matrix printer came lisa think wish attach pc manual told sort printer disguise anyone help manuals info codes send select fonts italics etc want write printer driver thanks advance stuart stuart munn dod university sky black edinburgh therefore god st mirren scotland supporter 031 031 fax god | macintosh lisa dot matrix parallel printer | macintosh lisa dot matrix printer | 
|5|article writes hearing endless debate read os better dos windows finally enought play couple different operating systems decided put two products head head test many fellow suggested however desire whatsoever use version wont really says ie run windows apps 2021 run windows apps 386 mode something larger windows apps | win | win nt |    

These five examples exemplify the two general kinds of examples I saw when looking at predictions on the training set. The first three (`1`, `2`, and `3`) are examples where the network is perfectly predicting the true headline, whereas the last two (`4` and `5`) are examples where the network is adding or missing a word.

Example `1` is a particularly interesting example, as it's true headline is the exact headline predicted in example `4` of the validation set examples. This is a clear example of the network overfitting to the training data. 

## Next Steps 

Moving forward, I'll be taking several steps to train a model that generalizes more effectively and that predicts more coherent, accurate headlines: 

1. Rework the input format to more closely resemble the paper 
    - Cap the vocabulary size at 40k words 
    - Include stopwords in the input text
    - Include unknown words in the input text (e.g. words not in the vocab)
2. Rework the network architecture to more closely resemble the paper
    - Make it an attention based model 
3. Obtain additional data (the current data set is rather small, about ~18k articles), or build in some form of data augmentation (or both)
4. Try out a couple of custom loss functions
