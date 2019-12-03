*Grab Function Trial

Ezen script egy példa arra, hogy hogyan lehetne megvalósítani az elképzelt **Grab** funkciót. Vagyis azt a funkciót, amivel majd a virtuális elemeket a prezentáló személy meg tudja majd fogni, mozgatni, odébb rakni.
A **Grab** funkció terveink szerint a [**Frame Differencing**](../FrameDifferencing)-el egybekötve fog működni. A **Frame Differencing** által kapott képekből és alakzatokból kapott információkat felhasználva meghatározható az "elkapás" mozdulata neurális háló segítségével.
A jelenlegi próba verzió (2019.12.03) csupán szemléltetésre használható. Az "elkapás" pillanata a *space* billentyű lenyomásával jelezhető a programnak. Ekkor megkezdi a trackelést a Lucas-Kanade módszer segítségével.
Felmerült az a gondolat is, hogy az elkapás utáni mozgatást, a vektormező segítségével, eredővektorok számításaival is meg lehetne valósítani. Ezt a gondolatot végül a [**Shift**](../Lucas-Kanade) funkció implementálásához használtam fel.

A Lucas-Kanade módszerről [**itt**](../Lucas-Kanade) található bővebb leírás és további script-ek.