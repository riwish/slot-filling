import os
import sys
import re
import pickle
from pprint import pprint
from requests import get
from bs4 import BeautifulSoup
from unidecode import unidecode

knowledge = {
    'inc.city': {'text': None},
    'inc.houseNumber': {'regex': [r"\d+[a-zA-Z]*$"]},
    'inc.streetName': {'text': None},
    'inc.postalCode': {'regex': [r"[1-9][0-9]{3}[\s]?[A-Za-z]{2}"]},
    'goo.brand': {'text': ['yaris']},
    #'goo.brandType': {'text': ['fieten', '23', '100', 'zitmaaier', 'herenbeurs', 'powerpack', 'ontvanger', 'femmes', '328i', 'cb1000ra', 'ligot', 'vogel', 'macan', 'fh', 'kangoo', 'thinkpade540', 'broban', 'pdm', 'acetyleen', 'mini', 'nationale', 'ts55ebq230v', 'kniptang', 'poolse', 'lx', 'telefoon', 'pounds', 'green', 'm3', 'leer', '750', 'ds4', 'z1', 'oostenrijk', 'klep', 'panamera', '307', 'reistas', 'sixpack', 'note4', 'straatsteen', 'm8', '48', 'stroomkabel', 'sleutelbos', 'hd', 'k750', 'schoolpas', 'kettingzaag', 'coop', 'kleingeld', 'b', 'recorder', 'corvette', '430d', 'r8', 'mate9', 'rvs', 'journey', 'prikapperaat', 'note', 'linde', 'j5', 'brisson', 'straatnaambord', 'xperiaz5', 'royaums', 'irbbpck86', 'prolife', 'stoeptegel', 'jazz', 'ignis', 'wals', '740d', 'riva', 'techniker', 'enveloppe', 'hoppers', 'p20', 'orco', '5x', 'lotion', '420i', 'k2', 'submariner', 'p700', 'andreaskruis', 'hangslot', 'bilford', 'lupo', 'huisleutel', '513cdi', 'am', 'up', 'iphone', 'zakmodel', 'betaalrekening', 'schakelarmband', 'pas', 'ex3212', 'lippenstift', 'citan', 'cooper', 'persoonskaart', 'schuifdeur', '330', 'barchetta', 'seizoenskaart', 'espace', 'isacura', 'pzp', 'a8', 'leidingen', 'l70', 'informatique', 'adereindhuls', 'electric', 'zonnebril', 'd7000', 'creditcard', 'g4', 'ducato', 'glazen', 'politie', '16g', 'edge7', 'flatscreen', '206', 'boorslijper', 'classic', 'fj3rsz1', 'klauwhamer', 'invalzaag', 'c25', 'elektra', 'fh12', 'stangen', '330d', 'applause', 'bosmaaier', 'betaalpassen', 'x102b', 'amsterdammertje', '525d', 'bibliotheek', 'port', 'u2716d', 'costelloe', 'clutch', 'pavillion', 'invalide', 'verschillende', 'apple', 'zorgverzekering', 'zkxqsz1', 'box', 'ziekenfonds', 'talisman', 'dizo', 'g6', 'spider', 'wertheim', 'terios', 'boorhamer', '7t', 'ux305', 'y5', 'oled55c8', '802', 'regiotaxi', '600', 'billfold', 'c5', 'walnoten', 'm187', 'fietssleutel', 'giftcard', 'mantel', 'xo', 'thema', 'brugleuning', '1m9', 'cadeaubonnen', 'hedrahekwerk', 'azijn', 'z240', 'binnentas', 'italiaans', 'aviator', 'hond', 'aveo', 'ohra', 'koperkabel', 'matiz', 'ix78p10h5', 'maxi', '1000', 'splash', 'c44', '7plus', 'h200', '2008', 'iwkb4p831', 'fabia', 'mate7', 'airpods', 'e6', 'filter', 'inhoud', '300c', '123', 'b07', 'insight', 'fs7', 'geen', 'dsw', 'munsterlander', 'h100', 'trafic', 'bonuskaart', 'cadeaubon', '220', 'achmea', 'xpro', 'limousine', 'autosleutel', 'hu05w', '525', 'ab', '63l053', '8900', 'belgie', 'kreukelzone', 'egg', 'autokluis', 'zundapp', 'sng355hn', 'ambition', 'ddf481', 'pasjeshouder', 'duitse', 'personenauto', 'molenhopper', 'pasfoto', '320ed', 'prada', 'ranger', '5c', 'flesjes', 'lt', 'ultimate', 'kitspuit', 'roomster', 'kopjes', 'powerhub', 'seijger', '360', 'varken', 'standspand', '306', 'rockburn', 'xvz13a', 'iphone6s', 'recipro', 'one', 'omega', 'c2', 'frame', '353316075057524', 'terrasomheining', '900', 'multivan', 'conservatorium', 'team', 'steeksleutels', 'verkadepas', 'slate21', 'wissel', 'lederen', '4171', 'nuc', 'stoppenkast', 'trottoir', 'klapbeurs', 't6', 'knip', 'octavia', 'impreza', 'ibiza', 'powerflush', 'ludix', 'fohn', 't460', 'cityped', 'g355', 'rolsteiger', 'laptop', 'p7', 'hk3770', '353555082010386', 'roazz', 'magnum', 'antec', '2', 'xt660x', 'airpod', 'herenmodel', 'laptops', 'pacifica', 'hollandpass', 'yachtmaster', 'biljetten', 'aluminium', 'rondmodel', 'provisional', '5se', 'fwvga', 'twister', '1phone6', 'traveller', 'ooiervaarspas', 'a40', '2670', 'klein', 'actros', '6plus', '1b', 'cuore', 'alto', 'c1', 'annoniem', 'parkeerkaart', 'kersenvlaai', 'kortingspas', '961', '120i', 'justy', 'superb', 'gripzakjes', '130i', '826', 'waterstof', 'x5xdrive30d', 'leon', 'israelische', 'tamahagane', 'klap', 'pavilion', 'autosleutels', 'pinautomaat', '15', 'europower', 'lantaarnpaal', 'kodiaq', 'sharan', 'paviljoen', 'modus', 'gsm', 'p8lite', 'j3', 'businesskaart', 'grappa', 'goldcard', 'zak', 'alpha', 'kaartje', 'empora', 'zandsteen', 'scenic', 'aed', 'robinmobile', 'nemo', 'lva0287', 'zegelring', 'meterkastdeur', 'canter', 'medaille', 'baardtrimmer', 'g3', 'tt', 'getz', 'roulettetafel', 'wafelijzer', 'etui', 'ampera', 'm135i', 'whiskey', 'a70', '6', 'mileupas', 'turkse', 'h6100bmclst', 'g13se2', 'kruidvat', 'eos', 'hek', 'mallorca', 'tuinmuurtje', 'r134a', 'plasmasnijder', 'ziekenfondspas', 'ssi', 'v40', 'desire', 'polo', 'campingstoel', 'boodschappen', 'kalos', 'notitieboek', 'kaart', 'az05w', 'iphonex', 'diversen', 'afzettingspalen', 'mountainbike', 'schoudertas', 'natwest', 'digitaal', 'punto', 'xr', 'accutol', 'altea', 'leren', 'gx160', 'heritage', 'bibliotheekpas', 'unitedconsumers', 'astra', 'forester', 'aktie', 'pt0305', 'twintig', 'dollars', 'landcruiser', 'informatiezuil', 'comfort', 'xps', 'g960f7ds', 'vivoactive', '5zjj68', 'labelwriter', 'p', 'rijst', 'yoximo', 'pilotentas', 'vitara', 'raak', 'spanje', 'citigo', 'malaysia', 'mensis', 'pv', 'tennaamgestelde', 'z2', 'koffiepads', 'kunstleer', 'steinerbos', 'mengtol', 'waarschuwlicht', '7470', 'ziekenhuis', 'rugtas', 'honingbier', '210', 'v50', 's6', 's8plus', 'tm48xs', 'handstuk', 'hema', 'oortjes', 'chocolademelk', 'nibhv', 'bankkaarten', 'skoolmate', 'contactloos', 'wakom', 'ring', 's5', 'wereldpas', 'sx710hs', 'ditzo', 't2', '26664783', 'c381', 'veloster', 'duinkaart', 'pasje', 'sirion', 'herenbril', 'handcomputer', 'professional', 'parrat', 'taiwanese', '508', 'damens', 'id', 'steel', 'koleos', '9700', 's80', 'wielerhulp', 'scala', 'probook', 'grote', 'p10', 'uitritzuil', '500', 'muntgeld', '735', 'kp3z', '700rv', 'pasjes', 'f7008s', 'q7', 'klopboormachine', 'tekentablet', 's5neo', 'colt', 'm57', 'tuinmuur', 'c4', 'vivaro', 'combihamer', 'wandarmatuur', 'orlando', 'betonhulpstof', 'menzis', 'commando', 'play', 'studentencard', 'anyroad', 'psr1440li', 'barner', 'bokshandschoen', 'i20', 'fusion', 'trend', 'imac', 'kitchenaid', '307d', 'bloembak', '08', 't50d', 'rapid', 'trolley', 'bouwradio', '22', 'ts410', '6t', '32ttsk', 'iak', 'studenten', 'creditkaart', 'ateca', 'qwerty', 'balr', 'mi', 'al', 'envelopmodel', 'boombox', 'bm18', 'divers', 'e532', 'kadjar', 'z24n', 'buiktasje', 'ts112', 'schaal', 'klapmodel', 'caddy', 'sandero', 'r580', 'winkel', 'aygo', 'pergula', 'forfour', 'bankbiljetten', 'slotpennen', 'blanco', 'balpen', 'xp500', 'cherokee', 'xperia', 'stoeprand', 'amplicomms', 'knipmes', 'megane', 'smartphone', 'onderdelen', 'raam', 'playstation', 'hoed', 'mokka', 'monza', 'motorfiets', 'fab', 'aquatimer', 'regenjas', 'motor', 'pb71305', 'lashandschoenen', 'contactgrill', 'rood', 'crealev', 'asfalt', 'oximo', 'alhambra', '530i', 'goudhaantje', 'prullenbak', 'iphone5', 'damesbeurs', 'pas215', 'elite', 'x7', 'mi3', '566', 'bonus', 'ispo', 'lidschapskaart', 'paris', 'leder', 'tegelsnijder', 'multimeter', 'bochtschild', 'beurs', 'passie', 'gra', 'diamand', 'golf', 'q3', 'speedfight', 'd3100', 'g5', 'rugzak', 'envelop', 'dollar', 'fs94rc', 'strakvlak', 'h512', 'vitality', 'babypoeder', 'leaderbord', 'benzine', 'sportschoenen', 'mii', 'a1', 'koffie', 'zendmachtiging', '412d', 'soldeerstation', 'gst90be', '7', 'reciprozaag', 'morfine', 'kassa', 'yaris', 'coolpix', 'chips', 'dcn692p2', 'weekendtas', 'tf', '640d', 'varkensrug', 'sx4', 'izacura', 'fontys', 'smart', 'anwb', 'panda', '2000', '316i', 'colmar', '318d', 'y6ii', 'dossier', 'viskist', 'doppendozen', '528i', 'bartscher', 'kisbee', 'mi2s', 'drukknoppaal', 'tupperware', 'ideapad', 'adam', 'tuinhek', 'agaponis', 'bipper', 'matrix', 'europool', 'primark', 'briefjes', 'zen', 'andumedic', 'roldeur', 'schouder', 'ford', 'a5100', 'm5', 'renegade', 'ing', '207', 'duiklamp', 'yeti', 'aquaracer', 'parka', 'clio', '4350fctj', 'c36', 'mondeo', 'metformin', 'bodywarmer', '518', 'passport', 'kabel', 'betaalpas', 'c9u', 'sportpas', 'sky', 'tugra', 'galaxys5', '3t', 'cruze', 'anonieme', 'bruin', '911e', 'debit', 'zeecontainer', 'g3s', 'v698bb', 'roldeurpoort', 'azivo', 'rechthoekig', 'rennsteig', 'crafter', 'vangrail', 'speedmaster', '308gtb', '24', 'cm600', 'yves', 'huo5w', 'pretfabriek', 'k756ua', 'tasje', 'atos', 'm64', 'xv535', 's40', 'nps50', 'x750l', 'spaarpas', 'diazepan', 'lisa', 'lightning', 'cdm800', 'dokker', 'cr85', 'p8', 'jumper', 'dx460', 'technoware', 'fiets', 'tundra', '325i', 'epica', 'lite', 'xcover', 'gokautomaat', 'glas', 'r730', 'dusseldorf', 'r410', 'bedrijfspand', 'imitatie', 'taxi', 'jack', 'hva', 'litouws', 'picnic', 'nc750xd', '8', 'movano', '93', '4', 'ds3', 'bioscoopbon', 'eurobord', 's6edge', 'd5200', 'elitebook', 'ts420', 'nederlands', 'tgx', 'viaduct', 'pathe', 'i10', '308', 'air', 'buitendeur', '320d', 'interpolis', '280', 'city', 'wodka', 'metrokaart', 'amarok', '1', 'stadspas', 'stofzuiger', 'amach', 'cirkelzaag', 'a3', 'flagship', 'g410', 'vespelini', 'motorjack', 'dcpl8400cdn', 't', 'micra', 'studentenpas', '516cdi', 'leuning', 'smg955f', 'hp450g3', 'uk', 'lichtmast', 'toegangsdeur', 'x1', 'unigarant', '900f', 's7', 'tz4', 'v90', '66lrnt', 'fiesta', 'cbr600rr', 'militair', 'gamecube', 'fb83b', '118i', 'max', 'j7', 'dschx90vb', 'beugelsluiting', 'z750', 'neo', 'omni100pc', 'muismat', 'airmax', 'vaio', 'c6', 'ooievaarspas', 'cz', 'water', 'laadapparaat', 'rimini', 'nederlandse', 'spaans', 'koki', 'tiguan', 'careyn', 'mastercard', 'russische', 'chronorally', 'x', '974', 'verlengkabel', 'viao', '35s13', 'thunderbolt', 'wallet', 'zieleman', 'macbookair', 'groen', 'vca', 'trek', 'secrid', 'master', 'parfum', '16', 'x360', 'vito', 'damesmodel', 'a', 'x3', 'fox', 'br600', 'edge', 'dames', 'scherf', '5t', 'dhr243', 'airstrap', '20', 'caliber', 'pijptang', 'keramisch', '5', 'd', 'cdj2000nxs', 'xs', 'draaipoort', 'i3', 'gf', 'butterfly', 'auto', 'deurslot', 'abarth', 'heg', 'logo', 'h2', '118d', '8plus', 'katalysators', 'ypsilon', 'p9', 'orient', 'handtasje', 'ds5', 'diverse', 'duits', 'ik', 'pixo', 'dasty', 'portemonee', 'gereedschap', '6p', 'dmr107', 'nld90230642', 'xc90', 'zaklamp', 'pools', 'longsleeve', 'voordeur', 'universiteit', 'rozen', 'normaal', '807', 'kadett', 'iza', 'jumpy', 's9plus', 'grand', 'haakseslijper', 'powerbank', 'e', 'desktop', 'prius', 'promovendum', 'tigra', 'crv', '523i', 'labelmanager', 'soulmate', 'ufj', 'xeperia', 's9600', '330i', 'fiddle', 'boor', 'roemeens', 'pet', 'valys', 'cilinder', 'ep6500te', '30', 'gevelplaat', 'achterdeur', 'duster', 'geldwisselaar', 'pinpas', 'i8', 'powershot', 'tweesteden', '6splus', 'hp255g3', 'stof', 'ioniq', 'europees', 'telefoonhoesje', 'ov', 'digitale', 'u2719d', 's8', 'karoq', '70', 'trouwringen', '4s', 'onderlade', 'spaarkas', 'djv182zj', 'l5', 'cc', 'schroeftol', 'euro', 'aa', '18v', 'staatslot', 'auris', 'rozenkrans', 'galaxcy', 'flespaal', '4200', 'securitaspas', 'pantalon', 'gaskraan', 'gmto', 'pincet', 'sundance', 'atego', 'br700', 'zx', 'amazon', 'hogeschool', 'langwerpig', 'note3', 'agila', 'benz', 'fooienpot', '900ss', 'plus', 'vijftig', 'nx7400', 'e1003', '3109', 'af', 'rekord', 'portemonnee', '7hk', 'achterwand', 'videotheek', 'durango', 'v641', 'museumjaarkaart', 'chip', 'smeedijzeren', 'sa14', '2610', '410d', 'dcp6600dn', 'klantenkaart', 'a0001', 'grieks', 'accuboormachine', 'ns', 'california', 'telescopische', 'peuken', 'mustang', 'a6', 'axa', 'schuur', 'suv', 'djm800', 'civic', 'meriva', 'collection', 'passagepas', 'note2', '5233135926', 'armband', 'fame', 'sc33', 'carport', 'sento', '4350t', 'afkortzaag', 'pin', 'cardslide', 'vervoerspas', 'vclic', 'koffieautomaat', '301', 'sf3000', '642910', 'xj6', 'coupe', 'nutrilon', 'marokkaans', 'creamer', 'verlengsnoeren', 'securitas', 'kuga', 'mi4c', 'bankbiljet', 'albatross', '0', 'amb', 'heren', 'xc40', 'ipad', 'mch', 'laguna', 'd5600', 'maatpak', 'groot', 'd5300', 'allianz', 'rj03', 'ix20', 'brommer', 'suburban', 'p9lite', 'buideltasje', 'basic', 'rum', 'beetle', 'magnetic', 'flyboard', 'ahob', 'cm900', 'voorhamer', 'vgz', 'ihone', 'sony', 'rcz', 'buskaarten', '320i', 'hoesje', 'damesportemonne', 'a7', 'iphone5s', 'duos', 'se', 'bewuzt', 'ctek', 'passenhouder', '114i', 'niro', 'abnamro', 'ax', 'parelcollier', 'safraan', 'cm700', 'wehkamp', 'coa', 'ovchippas', 'briefgeld', 'j530f', 'fysio', 'caravelle', 'sporttas', 's7s', 'transporter', 'touran', 'rooy', 'viano', '190sl', 'runner', 'galaxy6', 'beveiligingspas', 'intergraal', '85l', 'schuifpoort', 'q5', 'c70', 'boom', 'koffiepot', 'light', 'berlingo', 'dokterstas', 'struisvogelleer', 'k1ba001', 'k1300s', 'knipmodel', 'cordoba', 'zoe', 'jaarkaart', 'v320', 'poller', 'vodka', 'mac', 'focus', 'nvt', 'slagmoersleutel', '230', 'electrisch', 'r1100r', 'aspire', '108', 'dtd146', 'lumia', 'm4', 'pb1', 'v60', 'g7', 's9', 'slijptol', 'australisch', '2force', 'bijenkorf', '208', 'transit', 'cijferhangslot', 'd855', 'dubbelglas', 'wbt', 'berden', 'bouwplaats', 'sportage', 'baleno', 'vectra', 'krachtkabel', 'k9', 'x550dp', 'voorraadplank', 'zafira', '105', 'bataviastad', '318i', 'ixus', 'a4', 'dink', 'lodgy', '116ed', 'tassen', 'erdinger', 'sorento', 'madone', 'barcelana', 'oprit', 'ms200', 'katalysator', 'voedingskabel', 'c38', '56340', 'steenboren', 'moneycard', 'jumbo', 'sport', 'verhoging', 'tackers', 'overwegboom', 'iphone6', '116710ln', 'satellite', 'c3', 'vgr2535', 'woning', 'arosa', 'boekmodel', 'leiden', 'bivakmuts', 'graafschap', 'switch', 'lemon', 'metapace', 'businesscard', 'windjack', '2016', 'lasapparaat', 'bloedgroepkaart', 'connexion', 'portefeuille', 'personeelspas', 'vanizuil', 'persoonlijke', 'persoonlijk', 'geldkistje', 'huissleutel', 'partner', 'corsa', 'reisverzekering', 'slot', 'hr2811ft', 'trimmer', 'kaarthouder', 'waterkoker', 'wbk', 'hopper', 'doblo', 'matrijzen', 'captiva', 'gouden', 'longboard', 'picanto', '911', '520', 'touareg', 'spijkerbroek', 'j6', 'be', 'jetta', 'leesbril', '407', 'jaszak', '190', 'handtas', 'bankpassen', 'israelisch', 'dune', 'b54465', 'yrv', '407d', 'optiplex', 'insulinespuiten', 'ikl3j1jd8', 't3', 'crossfire', 'giropas', 'insignia', 'kizashi', '2500m', 'accent', 'boxer', 'x5', '290', '41', 'vu', '1234yf', 's2', 'halsketting', 'galaxy', 'captur', 'a150', 'biro', 'grondkabel', 's10', '51143573z', 'ziekenhuispas', 'gti', 'xt1622', '03txx6', 'sportief', 'nikon', 'venga', 'nitro', 'armani', 'legacy', 's4', '625', 'megger', 'j76', 'zorgpas', 'pakjes', 'curve', 'koningsketting', 'geldbedrag', 'lamellen', 'nexus', 's', 'huishoudmodel', '25', '10', 'disselslot', 'dameshorloge', '80d', 'plaat', 'ms280c', 'parkline', 'oplader', 'ascona', 'modeo', '3008', 'pc', '1a', '33q7', 'promax', 'belgisch', 'logan', 'pl', 'macbook', 'cr3v2', 'clips', 's60', '2018', 'veiligheidsglas', 'twoface', 'nijptang', '340i', '116i', 'amandelen', 'kortingskaart', 'antara', 'vivacity', 'bouwstofzuiger', 'pantein', 'sloophamer', 'ix35', 'beeld', 'rolex', 'sparkasse', 'amro', 'vouw', '013', 'robot', 'novamatic', 'bovenfrees', 'prepay', '4856257800', 'winterjas', 'celerio', 'onvz', 'stonic', 'vouwmodel', 'm28', 'a50', 'c46', '630', 'damestas', 'methylfenidaat', 'a1312', 'bromfiets', 'tucson', 'xc60', 'n700', 'dl1000', 'optima', 'x75vdty216h', 'stiftslijper', '1490', '120d', 'spark', 'experia', 'westpac', 'bandana', 'twingo', 'boortjes', 'beton', 'wegdek', 'pijnboompitten', 'f150', 'swift', 'labelprinter', 'kapex', 'gevelbord', 'scudo', '8n', 'lumix', '5s', 'carens', 'materia', 'move', 'frans', 'wegafzetting', 'bladblazer', 'c30', 'maandkaart', 'capture', 'museum', 's7560', 'staven', 'ka', 'senseo', 'combo', 'yprade', 'rugged', '5008', 'h5380bd', 'suede', 'isolatieglas', 's3', 'aerox', '35s14n', 'museumkaart', 'abonnement', 'ae47xs', '3', 'earpods', '144', 'd750', 'ip284', 'bouwcontainer', 'rio', '7200', 'expert', 'india', 'bankpas', 'openklap', 'kever', 'jeans', 'proceed', 'dremel', 'kanteldeur', 'g5se', 'h1', 'kassalade', 'v70', 'i40', '6s', 'd2', 'zilverenkruis', 'universiteitspa', 'kings', '50', 'evoque', 'adr', 'wijn', 'kombi', 'transport', 'rotex', 'rabobank', 'iphone10', 'istdl2', 'supra', 'k5', 'anoniem', 'bosrand', 'onbekend', 'scirocco', 'zakelijk', 'xc70', 'pashouder', 's1', 'huisdeur', 'barrier', 'trouwring', 'astro', 'bedrijsbus', 'chromebook', '520d', 'ruimers', 'y6', 'scooter', 'fz1', '418d', 'portomonnee', 'wegenwacht', 'bank', 'laser', 'n26', 'ipod', 'koevoet', 'sprinter', 'aiphone', 'a5', 'mega', 'leerachtig', 'akrapovic', 'festivalmunten', 'vierkant', 'e16', 'ux410u', 'te', 'passat', '11', 'rijschaaf', 'sq5', '530d', 'accu', 'h850', 'hardmetaal', 'heftruck', 'lszm', 'zip', 'silvia', 'huishoud', 'stuurslot', 'motorzaag', 'titanium', 'paspoort', 'ds', 'opwindhorloge', 'iphone7', 'sns', 'telpaal']},
    'goo.licensePlate': {'regex': [r"[a-zA-Z]{2}[\d]{2}[\d]{2}", r"[\d]{2}[\d]{2}[a-zA-Z]{2}", r"[\d]{2}[a-zA-Z]{2}[\d]{2}",
                                   r"[a-zA-Z]{2}[\d]{2}[a-zA-Z]{2}", r"[a-zA-Z]{2}[a-zA-Z]{2}[\d]{2}", r"[\d]{2}[a-zA-Z]{2}[a-zA-Z]{2}",
                                   r"[\d]{2}[a-zA-Z]{3}[\d]{1}", r"[\d]{1}[a-zA-Z]{3}[\d]{2}", r"[a-zA-Z]{2}[\d]{3}[a-zA-Z]{1}",
                                   r"[a-zA-Z]{1}[\d]{3}[a-zA-Z]{2}", r"[a-zA-Z]{3}[\d]{2}[a-zA-Z]{1}", r"[a-zA-Z]{1}[\d]{2}[a-zA-Z]{3}",
                                   r"[\d]{1}[a-zA-Z]{2}[\d]{3}", r"[\d]{3}[a-zA-Z]{2}[\d]{1}"]},
    'goo.mainColor': {'text': None},
    'goo.type': {'text': ['rijbewijs', 'paspoort', 'bankpas']},
    #'goo.type': {'text': ['bril', 'projector', 'lidmaatschap', 'auto', 'trilmachine', 'snacks', 'visakte', 'lens', 'rioolput', 'brons', 'toegangspas', 'parasol', 'buitenlands', 'klantenpas', 'rubberboot', 'creditcard', 'schoonmaakappa', 'kabel', 'monitor', 'hok', 'videocamera', 'plakband', 'sealbag', 'aggregaat', 'speelgoed', 'zalf', 'handgereedscha', 'topografie', 'alcohol', 'sleutel', 'handschoen', 'betaalautomaat', 'boek', 'bestek', 'studentenpas', 'stempel', 'kit', 'armband', 'stof', 'koffie', 'portemonnee', 'geld', 'gordijn', 'generator', 'zorgpas', 'luik', 'fiets', 'zonnescherm', 'idkaart', 'entreebewijs', 'vergunning', 'portefeuille', 'parfumerieen', 'tachograaf', 'schoolbenodigd', 'planten', 'grille', 'watermeter', 'toiletartikel', 'museum', 'tegels', 'vangrail', 'visartikel', 'overige', 'vlaggemast', 'golfartikel', 'riem', 'krat', 'kassa', 'wasapparatuur', 'trampoline', 'randapparatuur', 'visum', 'cd', 'ruit', 'spuitbus', 'zaad', 'doos', 'boren', 'keycard', 'frisdrank', 'goot', 'film', 'kabelkast', 'kinderzitje', 'velg', 'vuurwapen', 'hectometerpaal', 'toegangspoort', 'aluminium', 'hout', 'broche', 'hoofdtelefoon', 'pallet', 'verband', 'container', 'plattegrond', 'kast', 'geldlade', 'bedrijfswagen', 'waterstof', 'draad', 'struiken', 'draaitafel', 'sieraad', 'haspel', 'drukpers', 'waren', 'zink', 'isolatie', 'zwembad', 'nederlands', 'begrenzer', 'brievenbus', 'nikkel', 'telefoonhouder', 'verwarmingsapp', 'kleding', 'bromfiets', 'verkeersbord', 'tas', 'slagboom', 'weegapparatuur', 'stoel', 'tafelzilver', 'smartwatch', 'geldcassette', 'multitool', 'ahob', 'horloge', 'freesmachine', 'wiel', 'dvd', 'beautycase', 'bankbescheiden', 'filter', 'handboog', 'kist', 'deur', 'boormachine', 'olie', 'schuurmachine', 'sticker', 'bosmaaier', 'frame', 'telescoop', 'decoder', 'detector', 'water', 'klok', 'document', 'metaal', 'schaafmachine', 'pot', 'kantoorbenodig', 'vignet', 'raam', 'collegekaart', 'lood', 'fotocamera', 'bestuurderskrt', 'dakbedekking', 'filmcamera', 'tuingereedscha', 'mes', 'helm', 'map', 'stamper', 'steiger', 'emmer', 'videorecorder', 'spiegel', 'microfoon', 'masker', 'gasfles', 'telefoonkaart', 'paraplu', 'scanner', 'munten', 'verblijfsvergu', 'vitrine', 'afroombox', 'medicijn', 'portier', 'schilderij', 'slang', 'parkeerpas', 'bestrating', 'motorboot', 'thee', 'brandblusser', 'televisie', 'slijpmachine', 'vaarbewijs', 'hangslot', 'staal', 'toner', 'lettertang', 'microscoop', 'vogel', 'certificaat', 'beeld', 'mediaspeler', 'navigatiesyste', 'keuringsbewijs', 'landbouwvoertuig', 'rookwaar', 'monsterboekje', 'duikbrevet', 'bouwkraan', 'jalouzie', 'brug', 'afstandsbedien', 'jerrycan', 'vlees', 'vloerbedekking', 'mand', 'meterkast', 'piercing', 'papier', 'contentcard', 'zilver', 'snoep', 'belichting', 'buis', 'geluidswal', 'schroefmachine', 'ketting', 'gaas', 'sanitair', 'verzekeringspa', 'shirt', 'software', 'personenauto', 'lasapparatuur', 'schutting', 'vat', 'statief', 'draaibank', 'communicatieap', 'hekwerk', 'compressor', 'radiateur', 'verkeerszuil', 'fles', 'radio', 'toets', 'camera', 'lichtkoepel', 'kookapparatuur', 'claxon', 'hoofddeksel', 'kentekenbewijs', 'schrikhek', 'bureau', 'combinatie', 'zand', 'slot', 'sleutelbos', 'thermometer', 'luidspreker', 'tuinartikelen', 'band', 'landbouwgereed', 'muur', 'tablet', 'afvalbak', 'stuur', 'imperiaal', 'koper', 'verlichting', 'boordcomputer', 'ring', 'abri', 'foto', 'hoes', 'verfspuit', 'schoeisel', 'registratiebew', 'handboei', 'tafel', 'zaklantaarn', 'plantenbak', 'voedsel', 'ladder', 'luifel', 'hoogwerker', 'gehoorapparaat', 'pomp', 'vuilniszak', 'koffer', 'reclamebord', 'fust', 'computer', 'headset', 'kentekencard', 'bankrekening', 'ereader', 'pijp', 'verf', 'naaimachine', 'flitspaal', 'straatnaambord', 'waterscooter', 'airsoftwapen', 'rekenmachine', 'chauffeursdipl', 'reisdocument', 'bumper', 'drank', 'rolluik', 'plaatsnaambord', 'zegels', 'sleutels', 'relais', 'glas', 'messing', 'bomen', 'soundmixer', 'airbag', 'legitimatiebew', 'uitlaat', 'scherm', 'ijzer', 'zak', 'benzinepomp', 'drukwerk', 'beurs', 'zaagmachine', 'collectebus', 'stenen', 'kaartenautomaat', 'rooster', 'snaar', 'keukenartikel', 'rijbewijs', 'oorsieraad', 'steen', 'paspoort', 'overig', 'telefoon', 'acculader', 'begeleiderspas', 'zadel', 'kluis', 'bon', 'vlag', 'edelsteen', 'tankpas', 'carkit', 'euro', 'motor', 'mengpaneel', 'kozijn', 'sleutelhanger', 'benzine', 'spaarpot', 'motorfiets', 'vis', 'kompas', 'accu', 'zwaard', 'hanger', 'meetapparatuur', 'luchtdrukwapen', 'geldkist', 'kentekenplaat', 'brandkast', 'sloep', 'verzekeringspl', 'bouwlamp', 'huis', 'batterij', 'wild', 'diesel', 'bank', 'versterker', 'blikje', 'verkeerslicht']},
    'per.city': {'text': None},
    'per.familyName': {'text': None},
    'per.firstNames': {'text': None},
    'per.houseNumber': {'regex': [r"\d+[a-zA-Z]*$"]},
    'per.postalCode': {'regex': [r"[1-9][0-9]{3}[\s]?[A-Za-z]{2}"]},
    'per.streetName': {'text': None}
}


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_colornames():
    """
    We will use list of dutch companies from wikipedia + global fashion and car brands
    """
    print("------ colornames ------")
    collectables = []
    url = 'https://nl.wikipedia.org/wiki/Lijst_van_HTML-kleuren'
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    type(html_soup)
    content_raw = html_soup.find_all('td')

    content = []
    for con in content_raw:
        text = str(con.text).rstrip()
        if not any(x.isupper() for x in text):
            if len(text) > 1 and not represents_int(text) and not str(text).startswith('#') and not '%' in text:
                if '[' in text:
                    text = text.split('[')[0]
                # print(text)
                content.append(text)
    content = [e.lower() for e in content]
    content = content[:len(content)-4]
    # Update knowledge
    knowledge['goo.mainColor']['text'] = list(set(content))


def get_streetnames():
    """
    We will use list of dutch companies from wikipedia + global fashion and car brands
    """
    categories = {
        'streetnames': ['/wiki/Lijst_van_straten_in_Almelo', '/wiki/Lijst_van_straten_in_Amersfoort',
                        '/wiki/Lijst_van_straten_in_Amsterdam', '/wiki/Lijst_van_straten_in_Apeldoorn',
                        '/wiki/Lijst_van_straten_in_Assen', '/wiki/Lijst_van_straten_in_Baarn',
                        '/wiki/Lijst_van_straten_in_Blaricum', '/wiki/Lijst_van_straten_in_Bredevoort',
                        '/wiki/Lijst_van_straten_in_Bunschoten-Spakenburg',
                        '/wiki/Lijst_van_straten_in_Buren_(gemeente)',
                        '/wiki/Lijst_van_straten_in_Bussum', '/wiki/Lijst_van_straten_in_De_Bilt',
                        '/wiki/Lijst_van_straten_in_Delft', '/wiki/Lijst_van_straten_in_Deventer',
                        '/wiki/Lijst_van_straten_in_Doesburg', '/wiki/Lijst_van_straten_in_Doorn',
                        '/wiki/Lijst_van_straten_in_Eemnes', '/wiki/Lijst_van_straten_in_Geldermalsen',
                        '/wiki/Lijst_van_straten_in_Gemert-Bakel', '/wiki/Lijst_van_straten_in_Gouda',
                        '/wiki/Lijst_van_straten_in_Groningen_(stad)', '/wiki/Lijst_van_straten_in_Haarlem',
                        '/wiki/Lijst_van_straten_in_Hilversum', '/wiki/Lijst_van_straten_en_pleinen_in_Hoorn',
                        '/wiki/Lijst_van_straten_in_Huizen', '/wiki/Lijst_van_straten_in_Kampen',
                        '/wiki/Lijst_van_straten_in_Korendijk', '/wiki/Lijst_van_straten_in_Laren_(Noord-Holland)',
                        '/wiki/Lijst_van_straten_in_Leiden', '/wiki/Lijst_van_straten_in_Middelburg',
                        '/wiki/Lijst_van_straten_in_Naarden', '/wiki/Lijst_van_straten_in_Nijkerk',
                        '/wiki/Lijst_van_straten_in_Norg', '/wiki/Lijst_van_straten_in_Nunspeet',
                        '/wiki/Lijst_van_straten_in_gemeente_Olst-Wijhe', '/wiki/Lijst_van_straten_in_Putten',
                        '/wiki/Lijst_van_straten_in_Rhenen', '/wiki/Lijst_van_straten_in_Rotterdam',
                        '/wiki/Lijst_van_straten_in_Soest_(Nederland)', '/wiki/Lijst_van_straten_in_Stichtse_Vecht',
                        '/wiki/Lijst_van_straten_in_Tegelen', '/wiki/Lijst_van_straten_in_Terneuzen',
                        '/wiki/Lijst_van_straten_in_Utrecht_(stad)', '/wiki/Lijst_van_straten_in_Veenendaal',
                        '/wiki/Lijst_van_straten_in_gemeente_Voorst', '/wiki/Lijst_van_straten_in_Weesp',
                        '/wiki/Lijst_van_straten_in_Wijdemeren', '/wiki/Lijst_van_straten_in_Woerden',
                        '/wiki/Lijst_van_straten_in_Woudenberg', '/wiki/Lijst_van_straten_in_Zaltbommel',
                        '/wiki/Lijst_van_straten_in_Zeist', '/wiki/Lijst_van_straten_in_Zutphen'],
    }

    if not categories['streetnames']:
        # Collect urls first
        print("------ Collect URLS ------")
        collectables = []
        url = 'https://nl.wikipedia.org/wiki/Categorie:Lijsten_van_straten_en_pleinen_in_Nederland'
        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        type(html_soup)
        content_raw = html_soup.find_all('div', class_='mw-category-group')
        # print(content_raw)

        content = []
        for c in content_raw:
            for con in c.ul.find_all('li'):
                # print(con)
                text = str(con.a['href'])
                content.append(text)
        categories['streetnames'] = content

    # Collect and merge streetnames for all cities
    print("------ streetnames ------")
    collectables = []
    for url_suffix in categories['streetnames']:
        url = 'https://nl.wikipedia.org' + url_suffix
        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        content_raw = html_soup.find_all('div', class_='mw-parser-output_models')

        content = []
        for c in content_raw:
            for d in c.find_all('ul'):
                for con in d.find_all('li'):
                    # Prevents decoding issues later on
                    text = unidecode(con.text)
                    # Correct for noise
                    if re.match(re.compile(r'[0-9]\.{1,}'), text):
                        text = re.sub(re.compile(r'[0-9]\.{1,}'), '', text)
                    # Post-processing
                    if len(text.split(' ')) <= 4 and len(text) >= 5:
                        if '-' in text:
                            text = text.split('-')[0]
                        if '(' in text:
                            text = text.split('(')[0].rstrip()
                        content.append(text)
        content = [e.lower() for e in content]
        collectables += content

    # Update knowledge
    knowledge['inc.streetName']['text'] = list(set(collectables))
    knowledge['per.streetName']['text'] = list(set(collectables))


def get_cities():
    """
    We will use list of dutch companies from wikipedia + global fashion and car brands
    """
    categories = {'city': ["/wiki/Lijst_van_Nederlandse_plaatsen"]}

    for cat, urllist in categories.items():
        print("------ " + cat + " ------")
        collectables = []
        for url_suffix in urllist:
            url = 'https://nl.wikipedia.org' + url_suffix
            response = get(url)
            html_soup = BeautifulSoup(response.text, 'html.parser')
            type(html_soup)
            content_raw = html_soup.find_all('a')
            # print(content_raw)

            content = []
            for con in content_raw:
                text = con.text
                if '(' in text:
                    text = text.split(' ')[0]
                content.append(text)
            content = [e.lower() for e in content]
            collectables += content

        # Update knowledge
        knowledge['inc.city']['text'] = list(set(collectables[36:len(collectables) - 96]))
        knowledge['per.city']['text'] = knowledge['inc.city']['text']


def get_brands_and_names():
    """
    We will use list of dutch companies from wikipedia + global fashion and car brands
    """
    categories = {
        'brands': ["/wiki/Category:Dutch_brands", "/wiki/Category:Car_brands", "/wiki/Category:Clothing_brands"],
        'firstnames': ["/wiki/Category:Dutch_feminine_given_names", '/wiki/Category:Dutch_masculine_given_names'],
        'surnames': [
            "/wiki/Category:Dutch-language_surnames",
            "/w/index.php?title=Category:Dutch-language_surnames&pagefrom=De+Ligt%0ADe+Ligt#mw-pages",
            "/w/index.php?title=Category:Dutch-language_surnames&pagefrom=Huibers#mw-pages",
            "/w/index.php?title=Category:Dutch-language_surnames&pagefrom=Nelissen#mw-pages",
            "/w/index.php?title=Category:Dutch-language_surnames&pagefrom=Tellegen#mw-pages",
            "/w/index.php?title=Category:Dutch-language_surnames&pagefrom=Van+Eck#mw-pages",
            "/w/index.php?title=Category:Dutch-language_surnames&pagefrom=Vos+%28surname%29#mw-pages"
        ],
    }

    for cat, urllist in categories.items():
        print("------ " + cat + " ------")
        collectables = []
        for url_suffix in urllist:
            url = 'https://en.wikipedia.org' + url_suffix
            response = get(url)
            html_soup = BeautifulSoup(response.text, 'html.parser')
            type(html_soup)
            content_raw = html_soup.find_all('div', class_='mw-category-group')

            content = []
            for c in content_raw:
                for con in c.ul.find_all('li'):
                    if len(c) > 1:
                        text = con.a.text
                        if '(' in text:
                            text = text.split(' ')[0]
                        content.append(text)

            if url_suffix == '/wiki/Category:Dutch_brands':
                content = content[2::]
            elif url_suffix == '/wiki/Category:Car_brands':
                content = content[8::]
            elif cat == 'surnames':
                content = content[7::]
            content = [e.lower() for e in content]
            collectables += content

        # Update knowledge
        if cat == 'brands':
            knowledge['goo.brand']['text'] = collectables
        elif cat == 'firstnames':
            knowledge['per.firstNames']['text'] = collectables
        elif cat == 'surnames':
            knowledge['per.familyName']['text'] = collectables
        elif cat == 'city':
            knowledge['inc.city']['text'] = collectables


def save_pickle(object):
    with open(os.path.join(os.getcwd(), 'knowledge.pickle'), 'wb') as f:
        pickle.dump(object, f)


def load_pickle():
    with open(os.path.join(os.getcwd(), 'knowledge.pickle'), 'rb') as f:
        data = pickle.load(f)

    for k, v in data.items():
        print(k, v)
    exit()


def main():
    get_brands_and_names()
    get_cities()
    get_streetnames()
    get_colornames()
    save_pickle(knowledge)
    load_pickle()


if __name__ == '__main__':
    main()
