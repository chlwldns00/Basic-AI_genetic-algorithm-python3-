import math, random
import cv2  #opencv 라이브러리를 사용해서 실제 사진파일(지도사진) 에 좌표를 찍고 그위에 결과를 나타낼수 있게 하였다.

class City:                               #도시의 x,y좌표를 담은 class
    def __init__(self, x=None, y=None):
        self.x = None
        self.y = None
        if x is not None:
            self.x = x
        else:
            self.x = int(random.random() * 200)
        if y is not None:
            self.y = y
        else:
            self.y = int(random.random() * 200)
   
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceTo(self, city):                        #현재 도시와 도시 사이의 거리 반환 함수
        xDistance = abs(self.getX() - city.getX())
        yDistance = abs(self.getY() - city.getY())
        distance = math.sqrt( (xDistance*xDistance) + (yDistance*yDistance) )
        return distance

    def __repr__(self):
        return str(self.getX()) + ", " + str(self.getY())


class TourManager:   #chromosome class
    destinationCities = []       #chromosome(도시좌표 즉 도시이름 의 리스트)

    def addCity(self, city):   #setting chromosome func
        self.destinationCities.append(city)

    def getCity(self, index):
        return self.destinationCities[index]

    def numberOfCities(self):
        return len(self.destinationCities)


class Tour:         #도시와 도시 사이의 경로를 나타내는 class
    def __init__(self, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.tour = []
        self.fitness = 0.0   #적합도(shortest total distance)-> 1/distance=fitness
        self.distance = 0    #거리
        if tour is not None:
            self.tour = tour
        else:
            for i in range(0, self.tourmanager.numberOfCities()):
                self.tour.append(None)

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, index):
        return self.tour[index]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        geneString = 'Start -> '
        for i in range(0, self.tourSize()):
            geneString += str(self.getCity(i)) + ' -> '
        geneString += 'End'
        return geneString

    def generateIndividual(self):
        for cityIndex in range(0, self.tourmanager.numberOfCities()):
            self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
        random.shuffle(self.tour)

    def getCity(self, tourPosition):
        return self.tour[tourPosition]

    def setCity(self, tourPosition, city):
        self.tour[tourPosition] = city
        self.fitness = 0.0
        self.distance = 0

    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1/float(self.getDistance())
        return self.fitness

    def getDistance(self):             #
        if self.distance == 0:
            tourDistance = 0
            for cityIndex in range(0, self.tourSize()):
                fromCity = self.getCity(cityIndex)
                destinationCity = None
                if cityIndex+1 < self.tourSize():
                    destinationCity = self.getCity(cityIndex+1)
                else:
                    destinationCity = self.getCity(0)
                tourDistance += fromCity.distanceTo(destinationCity)
            self.distance = tourDistance
        return self.distance

    def tourSize(self):
        return len(self.tour)

    def containsCity(self, city):
        return city in self.tour


class Population:  #Tour들의 집합(도시들을 이은 선)을 나타내는 class              여러개의 tour집합의 fitness를 비교해서 fitness가 높은것들을 crossover & mutate 해서 더 좋은 population->tour집합을 만들어낸다.
    def __init__(self, tourmanager, populationSize, initialise):
        self.tours = []
        for i in range(0, populationSize):
            self.tours.append(None)
        
        if initialise:
            for i in range(0, populationSize):
                newTour = Tour(tourmanager)
                newTour.generateIndividual()
                self.saveTour(i, newTour)
        
    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    def saveTour(self, index, tour):
        self.tours[index] = tour

    def getTour(self, index):
        return self.tours[index]

    def getFittest(self):                            #가장높은 fitness를 가지는 population의 tour
        fittest = self.tours[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness() <= self.getTour(i).getFitness():
                fittest = self.getTour(i)
        return fittest

    def populationSize(self):
        return len(self.tours)


class GA:   #유전알고리즘 적용 class (돌연변이 비율 0.05프로,
    def __init__(self, tourmanager, mutationRate=0.05, tournamentSize=5, elitism=True):
        self.tourmanager = tourmanager
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.elitism = elitism

    def evolvePopulation(self, pop):  #진화(다음세대)하는 과정
        newPopulation = Population(self.tourmanager, pop.populationSize(), False)
        elitismOffset = 0
        if self.elitism:
            newPopulation.saveTour(0, pop.getFittest())
            elitismOffset = 1
        
        for i in range(elitismOffset, newPopulation.populationSize()):
            parent1 = self.tournamentSelection(pop)
            parent2 = self.tournamentSelection(pop)
            child = self.crossover(parent1, parent2)
            newPopulation.saveTour(i, child)
        
        for i in range(elitismOffset, newPopulation.populationSize()):
            self.mutate(newPopulation.getTour(i))
        
        return newPopulation
   
    def crossover(self, parent1, parent2): #partially matched crossover
        child = Tour(self.tourmanager)
        
        startPos = int(random.random() * parent1.tourSize())
        endPos = int(random.random() * parent1.tourSize())
        
        for i in range(0, child.tourSize()):
            if startPos < endPos and i > startPos and i < endPos:
                child.setCity(i, parent1.getCity(i))
            elif startPos > endPos:
                if not (i < startPos and i > endPos):
                    child.setCity(i, parent1.getCity(i))
        
        for i in range(0, parent2.tourSize()):
            if not child.containsCity(parent2.getCity(i)):
                for ii in range(0, child.tourSize()):
                    if child.getCity(ii) == None:
                        child.setCity(ii, parent2.getCity(i))
                        break

        return child
   
    def mutate(self, tour):         #부모->자식 으로 바뀌는 과정
        for tourPos1 in range(0, tour.tourSize()):
            if random.random() < self.mutationRate:
                tourPos2 = int(tour.tourSize() * random.random())
                
                city1 = tour.getCity(tourPos1)
                city2 = tour.getCity(tourPos2)
                
                tour.setCity(tourPos2, city1)
                tour.setCity(tourPos1, city2)

    def tournamentSelection(self, pop):
        tournament = Population(self.tourmanager, self.tournamentSize, False)
        for i in range(0, self.tournamentSize):
            randomId = int(random.random() * pop.populationSize())
            tournament.saveTour(i, pop.getTour(randomId))
        fittest = tournament.getFittest()
        return fittest


###################################################################################################  main  ###############################################################################################################################################


if __name__ == '__main__':

    n_cities = 20   #도시 갯수=모든도시의 리스트=chromosome
    population_size = 50 #tour의 갯수
    n_generations = 100  #부모 -> 자식세대 유전 알고리즘 적용횟수

    random.seed(200) #다른결과를 원하면 시드값을 계속바꿔주며 테스트해볼수있다.

    # Load the map
    map_original = cv2.imread('map.jpg')

    # Setup cities and tour
    tourmanager = TourManager()  

    for i in range(n_cities):       #랜덤한 죄표값으로 도시생성
        x = random.randint(200, 800)
        y = random.randint(200, 800)

        tourmanager.addCity(City(x=x, y=y)) #setting chromosome
        cv2.circle(map_original, center=(x, y), radius=10, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    cv2.imshow('map', map_original)
    cv2.waitKey(0)  #띄워진 map창을 끄면 다음코드가 실행됩니다.

    # Initialize population
    
    pop = Population(tourmanager, populationSize=population_size, initialise=True)  #처음에는 랜덤으로(initialise=True) population 생성후 가장높은 fitness를 보이는 population의 거리를 나타내어 준다.
    print("Initial distance: " + str(pop.getFittest().getDistance()))

    # Evolve population
    ga = GA(tourmanager)

    for i in range(n_generations):   #100세대의 유전알고리즘을 거쳐서 
        pop = ga.evolvePopulation(pop)

        fittest = pop.getFittest()

        map_result = map_original.copy()

        for j in range(1, n_cities):
            cv2.line(                                  #도시를 이은선 표시(경로=tour)
                map_result,
                pt1=(fittest[j-1].x, fittest[j-1].y),
                pt2=(fittest[j].x, fittest[j].y),
                color=(255, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA
            )

        cv2.putText(map_result, org=(10, 25), text='Generation: %d' % (i+1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=0, thickness=1, lineType=cv2.LINE_AA) #몇세대째 진행중인지 출력
        cv2.putText(map_result, org=(10, 50), text='Distance: %.2fkm' % fittest.getDistance(), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=0, thickness=1, lineType=cv2.LINE_AA) #가장높은 fitness를 가진 population의 거리 출력
        cv2.imshow('map', map_result)
        if cv2.waitKey(100) == ord('q'):
            break

    # Print final results
    print("Finished")
    print("Final distance: " + str(pop.getFittest().getDistance())) #세대 반복이 끝난후에 가장높은 fitness를 가진 population의 거리 출력
    print("Solution:")
    print(pop.getFittest()) #fittest population의 tour의 좌표값 나열

    cv2.waitKey(0)

