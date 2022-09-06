def filterXaiMethod(pool,value,filterReasons):
    newPool = filter(lambda x: value.filterComapre(x.getXaiMethod()), pool)
    filterReasons.append('Filtered XaiMethods='+value.getName())
    return list(newPool)

def filterTaskType(pool,value,filterReasons):
    newPool = filter(lambda x: value.filterComapre(x.getTaskType()), pool)
    filterReasons.append('Filtered TaskType='+value.getName())
    return list(newPool)

def filterScope(pool,value,filterReasons):
    newPool = filter(lambda x: value.filterComapre(x.getScope()), pool)
    filterReasons.append('Filtered Scope='+value.getName())
    return list(newPool)

def filterResultType(pool,value,filterReasons):
    newPool = filter(lambda x: value.filterComapre(x.getResults()), pool)
    filterReasons.append('Filtered ResultType='+value.getName())
    return list(newPool)

def rankPool(pool,properties):
    res={}
    for method in pool:
        methodCounter=0
        for prop, value in properties.items():
            rank=method.rank(prop,value)
            if(rank is not None):
                methodCounter+=rank
        if(str(methodCounter) not in res):
            res[str(methodCounter)]=method
        else:
            if(type(res[str(methodCounter)])!= list):
                res[str(methodCounter)]=[res[str(methodCounter)],method]
            else:
                res[str(methodCounter)].append(method)
    return res
