
GRAPHVIZ_COLORS = ['coral2', 'cadetblue3', 'gold', 'coral1', 'aquamarine4', 'darkslategrey',
                   'cyan2', 'antiquewhite3', 'coral', 'azure4', 'darkkhaki', 'deeppink',
                   'antiquewhite4', 'beige', 'firebrick3', 'firebrick1', 'brown1', 'darkgreen',
                   'cornsilk1', 'darkgoldenrod2', 'cornsilk2', 'bisque3', 'darkseagreen', 'darkgoldenrod1',
                   'deeppink3', 'darkviolet', 'chocolate2', 'crimson', 'goldenrod', 'darkseagreen2',
                   'cornflowerblue', 'blue2', 'aquamarine2', 'darkolivegreen1', 'antiquewhite1', 'brown',
                   'deepskyblue3', 'darkslateblue', 'dodgerblue2', 'burlywood4', 'floralwhite',
                   'burlywood3', 'blue', 'blue3', 'deeppink1', 'blue4', 'dodgerblue4', 'cyan4', 'chocolate1',
                   'aquamarine1', 'darkgoldenrod4', 'darkturquoise', 'darkorchid1', 'dimgray', 'cadetblue2',
                   'darkolivegreen', 'darkorchid3', 'cadetblue1', 'azure1', 'deeppink2', 'burlywood2', 'chartreuse3',
                   'gold3', 'darkseagreen4', 'dodgerblue3', 'firebrick4', 'blanchedalmond', 'cyan', 'darkslategray',
                   'gold2', 'dimgrey', 'gold4', 'goldenrod2', 'goldenrod1', 'burlywood', 'cadetblue', 'gainsboro',
                   'bisque', 'deepskyblue4', 'cyan1', 'coral4', 'darkorchid2', 'deepskyblue1',
                   'darkorange3', 'ghostwhite', 'bisque2', 'darkslategray2', 'darkolivegreen3', 'cadetblue4',
                   'deepskyblue', 'aquamarine3', 'cyan3', 'brown2', 'darkseagreen1', 'chocolate4',
                   'darkorange', 'darkorange1', 'blueviolet', 'chartreuse', 'antiquewhite2', 'gold1',
                   'cornsilk3', 'darkolivegreen2', 'chartreuse2', 'darkgoldenrod', 'brown3', 'bisque1',
                   'darksalmon', 'deeppink4', 'blue1', 'darkgoldenrod3', 'darkolivegreen4', 'aliceblue',
                   'burlywood1', 'darkslategray1', 'azure2', 'azure', 'darkorchid', 'dodgerblue',
                   'cornsilk', 'darkseagreen3', 'firebrick', 'chocolate', 'goldenrod3', 'coral3', 'firebrick2',
                   'deepskyblue2', 'bisque4', 'darkslategray4', 'darkorange2', 'azure3', 'brown4', 'chartreuse1',
                   'chartreuse4', 'dodgerblue1', 'cornsilk4', 'darkorchid4', 'forestgreen', 'chocolate3',
                   'antiquewhite', 'goldenrod4', 'darkslategray3', 'darkorange4', 'aquamarine']


def color_from_hash(obj):
    ''' derives an individual color from an object (as a string) that can be used for a graphviz-graph '''
    from hashlib import md5
    hc = int(md5(str(obj).encode("utf-8")).hexdigest(), 16)
    return GRAPHVIZ_COLORS[hc % len(GRAPHVIZ_COLORS)]
