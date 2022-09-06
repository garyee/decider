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
            rank=rank(prop,value,method)
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

def rank(prop,value,method):
    constraints=method.getConstraints()
    if(prop not in constraints):
        return 1
    if(prop=='heterogeneity'):
        return constraints['heterogeneity'].rank(value)
    if(prop=='col_count'):
        return constraints['col_count'].rank(value)
    if(prop=='corr_det'):
        return constraints['corr_det'].rank(value)
    if(prop=='multicollinearity'):
        return constraints['multicollinearity'].rank(value)
    if(prop=='linearity'):
        return constraints['linearity'].rank(value)
    if(prop=='monotonicity'):
        return constraints['monotonicity'].rank(value)
    if(prop=='interactivity'):
        return constraints['interactivity'].rank(value)
        
