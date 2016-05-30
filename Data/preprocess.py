
cmnt_read = open('cmnt_test.txt','r');
post_read = open('post_test.txt','r');

dictionary_file = open('./used/dictionary.txt','w');
cmnt_index = open('./used/cmnt_index.txt','w');
post_index = open('./used/post_index.txt','w');

dictionary = list();
dictionary.append('<None>');
dictionary.append('<UNK>');
dictionary.append('<END>');

i = 0;
while True:
    line = cmnt_read.readline();
    if not line:
        print "total ",i,"sentences";
        print "touch the end of file";
        break;
    i += 1;
    lline = line.split('\t');
    line = ' '.join(lline);
    lline = line.split(' ');
    del lline[-1];
    lline = [x for x in lline if x is not ''];
    temp = list();
    for x in lline:
        if x not in dictionary:
            dictionary.append(x);
        temp.append(str(dictionary.index(x) + 1));
    temp.append(str(dictionary.index('<END>')));
    temp.append('\n');
    cmnt_index.write(' '.join(temp));

i = 0;
while True:
    line = post_read.readline();
    if not line:
        print "total ",i,"sentences";
        print "touch the end of file";
        break;
    i += 1;
    lline = line.split('\t');
    line = ' '.join(lline);
    lline = line.split(' ');
    del lline[-1];
    lline = [x for x in lline if x is not ''];
    temp = list();
    for x in lline:
        if x not in dictionary:
            dictionary.append(x);
        temp.append(str(dictionary.index(x) + 1));
    temp.append(str(dictionary.index('<END>')));
    temp.append('\n');
    post_index.write(' '.join(temp));

for i in xrange(len(dictionary)):
    dictionary_file.write(dictionary[i] + '\n');

post_index.close()
cmnt_index.close();
dictionary_file.close();

